import random

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
