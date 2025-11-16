import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs

class LargeMarginSurvLoss(nn.Module):
    """
    Stable combined loss:
      - Discrete-time NLL on CDF (calibration path)
      - Pairwise logistic ranking on DOT product with additive margin (stable; no NaNs)
      - Angular-margin hinge on COSINE (ArcFace/CosFace spirit)
      - Mild monotonicity penalty on CDF

    Hard-coded, conservative defaults to avoid NaNs from batch 1.
    """
    def __init__(self,
                 rank_weight: float = 0.2,
                 ang_margin_weight: float = 0.1,
                 mono_weight: float = 0.05,
                 margin: float = 0.15,
                 sigma: float = 0.5,
                 eps: float = 1e-7):
        super().__init__()
        self.rank_weight = rank_weight
        self.ang_margin_weight = ang_margin_weight
        self.mono_weight = mono_weight
        self.margin = margin
        self.sigma = max(sigma, 1e-6)  # guard
        self.eps = eps

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        # Comparable if i failed strictly before j, or same bin but j censored; only count i if event
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    @staticmethod
    def nll_from_cdf(cdf: torch.Tensor, label: torch.Tensor, event: torch.Tensor, eps: float) -> torch.Tensor:
        B, T = cdf.shape
        prevF = torch.cat([cdf.new_zeros(B, 1), cdf[:, :-1]], dim=1)
        pmf = (cdf - prevF).clamp_min(eps)              # p_k >= eps
        surv_tail = (1.0 - cdf).clamp_min(eps)          # S(y) >= eps
        idx = torch.arange(B, device=cdf.device)

        py = pmf[idx, label].clamp_min(eps)             # p_y
        Sy = surv_tail[idx, label]                      # S(y)

        e = event.float() if event.dtype == torch.bool else event
        loss_e = -torch.log(py[e == 1]).mean() if (e == 1).any() else cdf.new_tensor(0.)
        loss_c = -torch.log(Sy[e == 0]).mean() if (e == 0).any() else cdf.new_tensor(0.)
        return loss_e + loss_c

    @staticmethod
    def mono_penalty(cdf: torch.Tensor) -> torch.Tensor:
        diffs = cdf[:, 1:] - cdf[:, :-1]
        return F.relu(-diffs).mean()

    def rank_logistic_on_dot(self, s_dot: torch.Tensor, label: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Stable pairwise logistic with additive margin on DOT product:
          L = mean_{(i,j) in R} softplus( - ( (s_i - s_j) - m ) / sigma )
        This avoids exp overflow and NaNs on the first batch.
        """
        R = self.pair_rank_mat(label, event)  # [B,B]
        if R.sum() < 1:
            return s_dot.new_tensor(0.)

        si = s_dot.view(-1, 1)
        sj = s_dot.view(1, -1)
        diff = (si - sj) - self.margin
        # Only compute where R==1 to avoid wasting memory; use masked mean
        loss_mat = F.softplus(-(diff / self.sigma))
        loss = (loss_mat * R).sum() / (R.sum() + 1e-6)
        return loss

    def angular_margin_hinge(self, s_cos: torch.Tensor, label: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        ArcFace/CosFace-style angular gap on COSINE:
          pen = ReLU( m - (cos_i - cos_j) ), averaged over comparable pairs.
        Keeps geometry well-separated without touching calibration logits.
        """
        R = self.pair_rank_mat(label, event)
        if R.sum() < 1:
            return s_cos.new_tensor(0.)

        si = s_cos.view(-1, 1)
        sj = s_cos.view(1, -1)
        gap = si - sj
        pen = F.relu(self.margin - gap)
        return (pen * R).sum() / (R.sum() + 1e-6)

    def forward(self, outputs, data):
        F_pred = outputs.cdf           # [B,T]
        s_cos  = outputs.risk_cos      # [B]
        s_dot  = outputs.risk          # [B] (now dot product)
        y      = data['label']
        e      = data['event']

        nll  = self.nll_from_cdf(F_pred, y, e, self.eps)
        rdot = self.rank_logistic_on_dot(s_dot, y, e)
        angp = self.angular_margin_hinge(s_cos, y, e)
        mono = self.mono_penalty(F_pred)

        return nll + self.rank_weight * rdot + self.ang_margin_weight * angp + self.mono_weight * mono


    
class AngSurv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder   = get_encoder(args)
        self.d_hid     = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes

        self.head = nn.Linear(self.d_hid, 1, bias=False)
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))

        # KEEP ONLY THIS: thresholds for calibration
        self.biases = nn.Parameter(torch.linspace(-1, 1, self.n_classes))

        self.temp   = nn.Linear(1, 1)                 # tau from radius input
        self.scaler = nn.Parameter(torch.tensor(2.0)) # global scale

        self._eps = 1e-7
        self.criterion = LargeMarginSurvLoss()

    @staticmethod
    def _tau_from_radius(r: torch.Tensor) -> torch.Tensor:
        # r is [B,1]; temp(r) returns [B,1]; softplus -> positive; clamp for stability
        return torch.clamp(F.softplus(r), min=1e-3, max=5.0)

    def forward(self, data):
        x = data['data']
        feats = self.encoder(x)                                     # [B,D]
        r = feats.norm(dim=1, keepdim=True).clamp_min(1e-6)         # [B,1]
        u = feats / r                                               # [B,D]
        w_hat = F.normalize(self.head.weight, dim=1)                # [1,D]

        # Scores
        s_cos = (u @ w_hat.t()).view(-1)                            # cosine
        s_dot = (feats @ w_hat.t()).view(-1)                        # dot = r * cos

        # Calibration logits: (s_dot + b_k) * scaler * tau
        tau_in = self.temp(r)                                       # [B,1]
        tau    = self._tau_from_radius(tau_in)                      # [B,1]
        logits = (s_dot.view(-1,1) + self.biases.view(1,-1)) * (self.scaler * tau)  # [B,T]
        cdf = torch.sigmoid(logits).clamp(self._eps, 1.0 - self._eps)
        surv = 1.0 - cdf

        if not self.training:
            cdf = torch.cummax(cdf, dim=1).values.clamp_(min=self._eps, max=1.0 - self._eps)
            surv = 1.0 - cdf

        return ModelOutputs(
            features=feats,
            logits=logits,
            cdf=cdf,
            surv=surv,
            risk=s_dot,          # <- ranking loss & reported scalar use DOT
            risk_cos=s_cos,      # <- for angular-margin regularizer & diagnostics
            biases=self.biases,
            projection_weight=w_hat.view(-1),
            radius=r.view(-1),
            tau=tau.view(-1),
        )

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)
