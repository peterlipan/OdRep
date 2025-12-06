import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class MonotoneISplineLink(nn.Module):
    """
    General cumulative-link transformation using I-spline-like basis:
        h(z) = sigmoid( alpha * z + beta * sum_m softplus(w_m) * I_m(z) )

    - I_m(z) are non-decreasing basis functions, precomputed on a grid.
    - softplus(w_m) >= 0 ensures the spline part is monotone in z.
    - alpha, beta control how close we are to plain logistic (PO).
      At init: alpha ~ 1, beta ~ 0 -> behaves like logit.
    """
    def __init__(
        self,
        num_basis: int = 8,
        grid_size: int = 200,
        z_min: float = -8.0,
        z_max: float = 8.0,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.grid_size = grid_size
        self.z_min = z_min
        self.z_max = z_max

        # Fixed grid over logits
        z_grid = torch.linspace(z_min, z_max, grid_size)  # [G]
        self.register_buffer("z_grid", z_grid)

        # Knot positions for hat (degree-1 B-spline) basis
        knots = torch.linspace(z_min, z_max, num_basis)
        self.register_buffer("knots", knots)  # [M]

        # Precompute I-spline-like basis I[g, m] on the grid
        I_grid = self._precompute_ispline_basis(z_grid, knots)  # [G, M]
        self.register_buffer("I_grid", I_grid)

        # Learnable spline weights (non-negative via softplus)
        self.raw_weights = nn.Parameter(torch.full((num_basis,), -4.0))
        # softplus(-4) ~ 0.018 -> very small contribution at init

        # Learnable scales for the linear (PO) and spline parts
        self.alpha = nn.Parameter(torch.tensor(.5))   # starts as logit
        self.beta  = nn.Parameter(torch.tensor(.5))   # no spline at init

        # Optional global bias (can be 0)
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _hat_basis(z: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        G = z.shape[0]
        M = knots.shape[0]
        z_exp = z.view(G, 1)
        k_exp = knots.view(1, M)

        d = torch.abs(z_exp - k_exp)
        if M > 1:
            dx = (knots[1] - knots[0]).item()
        else:
            dx = 1.0
        H = torch.clamp(1.0 - d / dx, min=0.0)

        # Normalize rowwise so sum_m H_m(z) ≈ 1
        H_sum = H.sum(dim=1, keepdim=True) + 1e-8
        H = H / H_sum
        return H

    def _precompute_ispline_basis(self, z_grid: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        H = self._hat_basis(z_grid, knots)          # [G, M]
        dz = z_grid[1] - z_grid[0]
        I = torch.cumsum(H * dz, dim=0)             # [G, M]

        # normalize each basis to [0, 1]
        I_max = I[-1, :].clone()
        I_max[I_max <= 0] = 1.0
        I = I / I_max.view(1, -1)
        return I

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Clamp logits to grid range
        z_clamped = z.clamp(self.z_min, self.z_max)

        orig_shape = z_clamped.shape
        z_flat = z_clamped.view(-1)  # [N]

        # Map z to grid indices
        G = self.grid_size
        z_min, z_max = self.z_min, self.z_max
        u = (z_flat - z_min) / (z_max - z_min + 1e-8) * (G - 1)
        u_clamped = u.clamp(0, G - 1 - 1e-6)

        idx0 = u_clamped.floor().long()
        idx1 = torch.clamp(idx0 + 1, max=G - 1)
        w1 = (u_clamped - idx0.float()).unsqueeze(-1)  # [N, 1]

        I0 = self.I_grid[idx0]    # [N, M]
        I1 = self.I_grid[idx1]    # [N, M]
        I_z = (1.0 - w1) * I0 + w1 * I1  # [N, M]

        # Non-negative spline weights, initially very small
        w_pos = F.softplus(self.raw_weights)  # [M] >= 0

        spline_term = torch.matmul(I_z, w_pos)  # [N]

        # Combined link: linear (logit) + spline deviation
        g_flat = self.bias + self.alpha * z_flat + self.beta * spline_term

        h_flat = torch.sigmoid(g_flat)
        return h_flat.view(orig_shape)



class CDFLoss(nn.Module):
    def __init__(self, sigma: float = 0.5, rank_weight: float = 0.5, mono_weight: float = 0.5,
                 margin: float = 0.3):
        super().__init__()
        self.sigma = sigma
        self.rank_weight = rank_weight
        self.mono_weight = mono_weight
        self.margin = margin          # additive margin

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    def rank_loss_on_risk(self, risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        si = risk.view(-1, 1)
        sj = risk.view(1, -1)
        diff_risk = si - sj - self.margin
        rank_mat = self.pair_rank_mat(durations, events)
        exponent = (-diff_risk / self.sigma).clamp(max=30) # prevent overflow
        loss = rank_mat * torch.exp(exponent)
        return loss.sum() / (rank_mat.sum() + 1e-6)

    @staticmethod
    def nll_from_cdf(cdf, label, event, eps=1e-7):
        B, T = cdf.shape
        prevF = torch.cat([cdf.new_zeros(B, 1), cdf[:, :-1]], dim=1)
        pmf = (cdf - prevF).clamp_min(eps)        # p_k >= 0
        surv_tail = (1.0 - cdf).clamp_min(eps)    # S(y) >= 0
        idx = torch.arange(B, device=cdf.device)

        py = pmf[idx, label].clamp_min(eps)       # p_y
        Sy = surv_tail[idx, label]                # S(y)

        loss_e = -torch.log(py[event == 1]).mean() if (event == 1).any() else 0.0
        loss_c = -torch.log(Sy[event == 0]).mean() if (event == 0).any() else 0.0
        return loss_e + loss_c
    
    def monotonicity_loss(self, cdf: torch.Tensor) -> torch.Tensor:
        diffs = cdf[:, 1:] - cdf[:, :-1]
        violations = F.relu(-diffs)
        return violations.mean()


    def forward(self, outputs, data):
        F_pred = outputs.cdf
        risk   = outputs.risk
        label  = data['label']
        event  = data['event']
        duration = data['duration']

        nll   = self.nll_from_cdf(F_pred, label, event)
        rloss = self.rank_loss_on_risk(risk, duration, event)
        mloss = self.monotonicity_loss(F_pred)
        return nll + self.rank_weight * rloss + self.mono_weight * mloss

class OrdSurv(nn.Module):
    def __init__(self, args, link='po', eps=1e-4):
        super(OrdSurv, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1, bias=False)
        self.criterion = CDFLoss()
        self.scaler = nn.Parameter(torch.ones(1))  # Scale for logits
        self.link = link
        self.eps = eps

        if link == 'ispline':
            self.activation = MonotoneISplineLink()

        self.biases = self.init_biases()

    def init_biases(self):
        t = torch.linspace(self.eps, 1.0 - self.eps, self.n_classes)
        if self.link == 'ph':      # cloglog
            lam = 1.0
            init = torch.log(lam * t)        # approx log cumulative hazard
        elif self.link == 'po':    # logit
            init = torch.log(t / (1.0 - t))  # logit(t)
        elif self.link == 'probit':
            normal = torch.distributions.Normal(0., 1.)
            init = normal.icdf(t)
        elif self.link == 'ispline':
            init = torch.log(t / (1.0 - t))
        else:
            init = torch.linspace(-1, 1, self.n_classes)
        return nn.Parameter(init, requires_grad=True)

    def activate(self, logits):
        if self.link == 'po':
            return torch.sigmoid(logits)
        elif self.link == 'ph':
            return 1. - torch.exp(-torch.exp(logits))
        elif self.link == 'pro':
            normal = torch.distributions.Normal(0., 1.)
            return normal.cdf(logits)
        elif self.link == 'ispline':
            return self.activation(logits)
        else:
            raise ValueError(f"Unknown link function: {self.link}")
        
    def forward(self, data):

        features = self.encoder(data['data'])
        proj = self.head(features)  # [B, 1]
        logits = proj + self.biases.view(1, -1)  # [B, T]
        cdf = self.activate(self.scaler * logits)  # [B, T]
        risk = proj.view(-1)  # [B * T]

        # force monotonicity only during eval
        if not self.training:
            cdf = torch.cummax(cdf, dim=1).values.clamp_(min=self.eps, max=1.0 - self.eps)

        surv = 1. - cdf

        return ModelOutputs(features=features,
                            logits=logits,
                            cdf=cdf,
                            risk=risk,
                            surv=surv,
                            biases=self.biases,
                            projection_weight=self.head.weight.view(-1))

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)
