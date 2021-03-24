import random

from PIL import Image
from torchvision.transforms import functional


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
