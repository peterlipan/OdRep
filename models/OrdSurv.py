import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, 
                 cdf_weight: float = 1.,
                 monotonic_weight: float = 0.1,
                 sigma: float = 0.5,
                 margin: float = 0.1,
                 rank_weight: float = 0.1,
                 gamma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.cdf_weight = cdf_weight
        self.monotonic_weight = monotonic_weight
        self.margin = margin
        self.rank_weight = rank_weight
        self.gamma = gamma

    def rank_loss_on_score(self, score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label_i = label.view(-1, 1)
        label_j = label.view(1, -1)
        rank_mat = (label_i < label_j).float()
        diff_score = score.view(-1, 1) - score.view(1, -1)
        loss = rank_mat * torch.exp(-diff_score / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)

    def forward(self, outputs, data):
        F_pred = outputs.cdf
        score = outputs.risk

        label = data['label']

        _, C = F_pred.shape # C: number of classes
        device = F_pred.device

        label_idx = torch.arange(C, device=device).view(1, -1)
        label_exp = label.view(-1, 1)
        target = (label_idx >= label_exp).float()
        distance = (label_idx - label_exp).abs().float()
        decay_weight = torch.exp(-self.gamma * distance)

        cdf_loss = ((F_pred - target) ** 2 * decay_weight).sum() / (decay_weight.sum() + 1e-6)

        monotonic_penalty = F.relu(F_pred[:, :-1] - F_pred[:, 1:] + self.margin).mean()

        rank_loss = self.rank_loss_on_score(score, label)

        total_loss = (
            self.cdf_weight * cdf_loss +
            self.monotonic_weight * monotonic_penalty +
            self.rank_weight * rank_loss
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
