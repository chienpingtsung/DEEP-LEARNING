from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset


class ImageMaskFolder(Dataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        The directory should be organized as following tree, and match each stem of images and masks.
        .
        ├── images
        │   ├── 0.png
        │   ├── 1.png
        │   ├── 2.png
        │   └── ...
        └── masks
            ├── 0.png
            ├── 1.png
            ├── 2.png
            └── ...
        """
        super(ImageMaskFolder, self).__init__()

        self.image_path = Path(root).joinpath('image/')
        self.mask_path = Path(root).joinpath('mask/')

        image_stems = {p.stem for p in self.image_path.glob('*.png')}
        mask_stems = {p.stem for p in self.mask_path.glob('*.png')}
        assert not image_stems - mask_stems, f'Missing masks of images: {image_stems - mask_stems}.'
        assert not mask_stems - image_stems, f'Missing images of masks: {mask_stems - image_stems}.'
        self.stems = list(image_stems)

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        stem = self.stems[item]
        image = Image.open(self.image_path.joinpath(f'{stem}.png'))
        mask = Image.open(self.mask_path.joinpath(f'{stem}.png'))
        size = image.size

        if self.transforms:
            image, mask = self.transforms(image, mask)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask, stem, size


class ImageFolder(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None):
        super(ImageFolder, self).__init__()

        self.image_path = Path(root)

        self.stems = list(p.stem for p in self.image_path.glob('*.png'))

        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        stem = self.stems[item]
        image = Image.open(self.image_path.joinpath(f'{stem}.png'))
        size = image.size

        if self.transform:
            image = self.transform(image)

        return image, stem, size
