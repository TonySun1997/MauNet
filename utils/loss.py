import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Binary Focal Loss operating on **logits** (includes sigmoid internally).

    Parameters
    ----------
    alpha : float
        Weight for the positive (foreground) class.  >0.5 biases toward recall.
    gamma : float
        Focusing parameter.  Higher values down-weight easy examples more.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        # p_t = p where target=1, else 1-p
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss operating on **probabilities** (apply sigmoid before calling).

    Parameters
    ----------
    alpha : float
        Weight for false positives.
    beta : float
        Weight for false negatives.  beta > alpha biases toward recall.
    smooth : float
        Smoothing constant.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        fp = (inputs * (1.0 - targets)).sum()
        fn = ((1.0 - inputs) * targets).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CenterNetFocalLoss(nn.Module):
    """CenterNet-style Gaussian-heatmap focal loss (Zhou et al., 2019).

    The target heatmap ``y ∈ [0, 1]`` is Gaussian-rendered at particle centers.
    Pixels with ``y == 1`` are positives; all others are negatives with a soft
    weighting ``(1 - y)^beta`` that down-weights pixels near a true center
    (they are "partial positives", not hard negatives).

    .. math::
        L_{pos} = -(1 - p)^{\\alpha}\\, \\log p                 \\quad (y = 1)
        L_{neg} = -(1 - y)^{\\beta}\\, p^{\\alpha}\\, \\log(1-p) \\cdot w_{neg}  \\quad (y < 1)
        L = (\\sum L_{pos} + \\sum L_{neg}) / \\max(N_{pos}, 1)

    Parameters
    ----------
    alpha : float
        Focusing exponent on the predicted probability (default 2.0).
    beta : float
        Exponent on ``(1 - y)`` that softens negative loss near true centers
        (default 4.0).
    eps : float
        Numerical clamp for log.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        neg_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        pos_mask = target.eq(1.0).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = -((1.0 - p) ** self.alpha) * torch.log(p) * pos_mask
        neg_loss = (
            -((1.0 - target) ** self.beta)
            * (p ** self.alpha)
            * torch.log(1.0 - p)
            * neg_mask
        )
        if neg_weight is not None:
            neg_loss = neg_loss * neg_weight

        n_pos = pos_mask.sum().clamp(min=1.0)
        return (pos_loss.sum() + neg_loss.sum()) / n_pos
