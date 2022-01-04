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


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            return tuple(functional.vflip(img) for img in args)
