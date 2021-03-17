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
