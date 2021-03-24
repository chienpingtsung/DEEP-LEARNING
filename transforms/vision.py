import random
from typing import Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional
from torchvision.transforms.transforms import _setup_size


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, *args):
        rst = []
        for pic in args:
            rst.append(functional.resize(pic, self.size, self.interpolation))
        return rst


class RandomRotation:
    def __init__(self, min, max, resample=False, expand=False, center=None, fill=None):
        self.min = min
        self.max = max
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, *args):
        angle = random.uniform(self.min, self.max)

        rst = []
        for pic in args:
            rst.append(functional.rotate(pic, angle, self.resample, self.expand, self.center, self.fill))
        return rst


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if torch.rand(1) < self.p:
            rst = []
            for pic in args:
                rst.append(functional.hflip(pic))
            return rst
        return args


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if torch.rand(1) < self.p:
            rst = []
            for pic in args:
                rst.append(functional.vflip(pic))
            return rst
        return args


class RandomCrop:
    def __init__(self, size):
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = functional._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self, *args):
        image, *_ = args
        i, j, h, w = self.get_params(image, self.size)

        rst = []
        for pic in args:
            rst.append(functional.crop(pic, i, j, h, w))
        return rst
