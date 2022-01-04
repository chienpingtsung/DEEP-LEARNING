import torch
from torch import nn, Tensor


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: (B, 1, H, W)
        :param target: (B, 1, H, W)
        """
        input = torch.sigmoid(input)
        B, *_ = target.shape

        input = input.view(B, -1)
        target = target.view(B, -1)

        intersection = input * target
        dice_coefficient = (2 * intersection.sum(axis=1) + 1) / (input.sum(axis=1) + target.sum(axis=1) + 1)

        return 1 - dice_coefficient.mean()
