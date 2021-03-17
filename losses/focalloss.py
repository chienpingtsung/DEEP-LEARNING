import torch
from torch import nn, Tensor
from torch.nn import functional


class FocalLoss(nn.Module):
    """
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: (B, 1, H, W)
        :param target: (B, 1, H, W)
        """
        p = torch.sigmoid(input)
        bce = functional.binary_cross_entropy(p, target, reduction='none')
        p_t = target * p + (1 - target) * (1 - p)
        loss = bce * ((1 - p_t) ** self.gamma)

        if self.alpha:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = loss * alpha_t

        return loss.mean()
