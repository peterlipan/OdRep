import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, 
                 cdf_weight: float = 1.,
                 sigma: float = 0.5,
                 rank_weight: float = 0.1,
                 cal_weight: float = 0.1,
                 gamma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.cdf_weight = cdf_weight
        self.rank_weight = rank_weight
        self.cal_weight = cal_weight
        self.gamma = gamma

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    def rank_loss_on_risk(self, risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        diff_risk = risk.view(-1, 1) - risk.view(1, -1)
        rank_mat = self.pair_rank_mat(durations, events)
        loss = rank_mat * torch.exp(-diff_risk / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)

    def calibration_loss(self, cdf: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        Brier-style calibration: compare predicted event probability with observed event.
        """
        # Predicted marginal event probability = final CDF value (prob(event before end of horizon))
        p_event = cdf[:, -1]   # [B]
        target = events.float()  # 1 if uncensored (event occurred), 0 if censored
        return ((p_event - target) ** 2).mean()

    def forward(self, outputs, data):
        F_pred = outputs.cdf
        risk = outputs.risk
        label = data['label']
        event = data['event']

        _, T = F_pred.shape
        device = F_pred.device

        time_idx = torch.arange(T, device=device).view(1, -1)
        label_exp = label.view(-1, 1)
        target = (time_idx >= label_exp).float()
        distance = (time_idx - label_exp).abs().float()
        decay_weight = torch.exp(-self.gamma * distance)

        mask = torch.where(event.view(-1, 1).bool(),
                           torch.ones_like(target).bool(),
                           (time_idx <= label_exp))

        decay_weight = decay_weight * mask.float()
        cdf_loss = ((F_pred - target) ** 2 * decay_weight)[mask].sum() / (decay_weight[mask].sum() + 1e-6)

        rank_loss = self.rank_loss_on_risk(risk, label, event)

        cal_loss = self.calibration_loss(F_pred, event)

        total_loss = (
            self.cdf_weight * cdf_loss +
            self.rank_weight * rank_loss +
            self.cal_weight * cal_loss
        )

        return total_loss


class OrdSurv(nn.Module):
    def __init__(self, args):
        super(OrdSurv, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1, bias=False)

        self.biases = nn.Parameter(torch.linspace(-1, 1, self.n_classes), requires_grad=True)

        self.criterion = CDFLoss()
        self.scaler = nn.Parameter(2. * torch.ones(1))  # Scale for logits

    def forward(self, data):

        features = self.encoder(data['data'])
        proj = self.head(features)  # [B, 1]
        logits = proj + self.biases.view(1, -1)  # [B, T]
        cdf = torch.sigmoid(logits * self.scaler)  # [B, T]
        risk = proj.view(-1)  # [B * T]
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

    def project_2d(self, data):
        features = self.encoder(data['data'])
        features = F.normalize(features, dim=1, p=2)
        proj = self.head(features)

        w = self.head.weight.squeeze(0)
        w = F.normalize(w, dim=0, p=2)
        v = torch.randn_like(w)
        v = v - (v @ w) * w
        v = F.normalize(v, dim=0, p=2)

        proj_x = features @ w
        proj_y = features @ v

        return ModelOutputs(x=proj_x, y=proj_y)
