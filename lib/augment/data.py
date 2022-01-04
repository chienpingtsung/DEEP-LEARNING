from torchvision.transforms import functional


class ToTensor:
    def __call__(self, *args):
        return tuple(functional.to_tensor(pic) for pic in args)
