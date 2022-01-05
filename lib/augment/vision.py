import random

from PIL import ImageFilter
from torchvision.transforms import functional


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, *args):
        image, *_ = args
        w, h = image.size
        tw, th = self.size
        top = random.randint(0, h - th if h - th > 0 else 0)
        left = random.randint(0, w - tw if w - tw > 0 else 0)
        return tuple(functional.crop(img, top, left, th, tw) for img in args)


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, *args):
        angle = random.choice(self.degrees)
        return tuple(functional.rotate(img, angle, expand=True) for img in args)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            return tuple(functional.hflip(img) for img in args)
        return args


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            return tuple(functional.vflip(img) for img in args)
        return args


class DivisiblePad:
    def __int__(self, base=16):
        self.base = base

    def __call__(self, img):
        w, h = img.size
        right = self.base - w % self.base if w % self.base else 0
        bottom = self.base - h % self.base if h % self.base else 0
        return functional.pad(img, [0, 0, right, bottom])


class Dilation:
    def __init__(self, size=3):
        self.size = size

    def __call__(self, img):
        return img.filter(ImageFilter.MaxFilter(self.size))
