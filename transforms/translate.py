from torchvision.transforms import functional


class ToTensor:
    def __call__(self, *args):
        rst = []
        for pic in args:
            rst.append(functional.to_tensor(pic))
        return rst
    