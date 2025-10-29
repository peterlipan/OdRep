import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, sigma: float = 0.5, rank_weight: float = 0.1,
                 margin: float = 0.15, crps_weight: float = 0.05, beta: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.rank_weight = rank_weight
        self.margin = margin          # cosine-style subtractive margin
        self.crps_weight = crps_weight
        self.beta = beta              # Laplace kernel bandwidth (in bins)
        self._eps = 1e-7

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    def rank_loss_on_risk(self, risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        # risk is ANGULAR (cosine) per your last change
        s_i = risk.view(-1, 1)
        s_j = risk.view(1, -1)
        rank_mat = self.pair_rank_mat(durations, events)
        # cosine margin: (s_i - m) - s_j
        diff = (s_i - self.margin) - s_j
        loss = rank_mat * torch.exp(-diff / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)

    @staticmethod
    def nll_from_cdf(cdf, label, event, eps=1e-7):
        B, T = cdf.shape
        prevF = torch.cat([cdf.new_zeros(B, 1), cdf[:, :-1]], dim=1)
        pmf = (cdf - prevF).clamp_min(eps)      # ΔCDF >= 0
        surv_tail = (1.0 - cdf).clamp_min(eps)  # S(y) >= 0
        idx = torch.arange(B, device=cdf.device)

        py = pmf[idx, label].clamp_min(eps)     # p_y
        Sy = surv_tail[idx, label]              # S(y)

        loss_e = -torch.log(py[event == 1]).mean() if (event == 1).any() else 0.0
        loss_c = -torch.log(Sy[event == 0]).mean() if (event == 0).any() else 0.0
        return loss_e + loss_c

    def crps_laplace_uniform(self, cdf: torch.Tensor, label: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Laplace-kernel CRPS on a UNIFORM bin grid (index distance).
        Weights w_{ik} ∝ exp(-|k - y_i| / beta), normalized per sample.
        Event target: step 1[k >= y]; Censored target: 0 for k <= y (conservative).
        """
        B, T = cdf.shape
        device = cdf.device
        idx = torch.arange(B, device=device)
        y = label
        e = event.float() if event.dtype == torch.bool else event

        # Distances |k - y_i|
        ar = torch.arange(T, device=device).view(1, T).expand(B, T)
        dist = (ar - y.view(-1, 1)).abs().float()

        # Laplace weights (Δt = 1 for uniform bins)
        w = torch.exp(-dist / self.beta)

        # Mask: events -> all bins; censored -> k <= y
        mask_cens = (ar <= y.view(-1, 1)).float()
        mask = torch.where(e.view(-1, 1) > 0.5, torch.ones_like(w), mask_cens)

        # Normalize per sample on active support
        w = w * mask
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(self._eps)
        w = w / w_sum

        # Targets
        step = (ar >= y.view(-1, 1)).float()
        target = torch.where(e.view(-1, 1) > 0.5, step, torch.zeros_like(step))

        crps = ((cdf - target) ** 2 * w).sum(dim=1).mean()
        return crps

    def forward(self, outputs, data):
        cdf   = outputs.cdf
        risk  = outputs.risk      # ANGULAR cosine score (from your updated forward)
        label = data['label']
        event = data['event']

        nll   = self.nll_from_cdf(cdf, label, event)
        rloss = self.rank_loss_on_risk(risk, label, event)
        crps  = self.crps_laplace_uniform(cdf, label, event)

        return nll + self.rank_weight * rloss + self.crps_weight * crps



class AngSurv(nn.Module):
    def __init__(self, args):
        super(AngSurv, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1, bias=False)

        self.biases = nn.Parameter(torch.linspace(-1, 1, self.n_classes), requires_grad=True)

        self.criterion = CDFLoss()
        self.scaler = nn.Parameter(2. * torch.ones(1))  # Scale for logits

    def forward(self, data):
        x = data['data']

        # Encode
        features = self.encoder(x)  # [B, D]

        # --- calibration path (dot product with raw head) ---
        proj = self.head(features)  # [B, 1]  uses raw weight -> keeps magnitude for NLL/CRPS
        logits = proj + self.biases.view(1, -1)  # [B, T]
        cdf = torch.sigmoid(logits * self.scaler).clamp(1e-7, 1 - 1e-7)
        surv = 1. - cdf

        # --- ranking path (angular / cosine) ---
        w_hat = F.normalize(self.head.weight, dim=1)          # [1, D], unit head
        radius = features.norm(dim=1, keepdim=True).clamp_min(1e-6)
        u = features / radius.detach()                        # unit direction; stop-grad on radius
        s_ang = (u @ w_hat.t()).view(-1)                      # [B], cosine-based risk in [-1,1]

        return ModelOutputs(features=features,
                            logits=logits,
                            cdf=cdf,
                            risk=s_ang,                       # <-- angular score for ranking
                            surv=surv,
                            biases=self.biases,
                            projection_weight=self.head.weight.view(-1),
                            radius=radius.view(-1))


    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)
