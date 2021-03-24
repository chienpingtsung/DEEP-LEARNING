from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset


class MaskFolder(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None):
        """
        The directory should be organized as following tree, and match each stem of images and labels.
        .
        ├── images
        │   ├── 0.png
        │   ├── 1.png
        │   ├── 2.png
        │   └── ...
        └── labels
            ├── 0.png
            ├── 1.png
            ├── 2.png
            └── ...

        :param root: Path to the dataset.
        :param transform: Transforms for data augmentation.
        """
        super(MaskFolder, self).__init__()

        self.image_path = Path(root).joinpath('images/')
        self.label_path = Path(root).joinpath('labels/')

        self.stems = {p.stem for p in self.image_path.glob('*.png')}
        assert not self.stems ^ {p.stem for p in self.label_path.glob('*.png')}, \
            'Missing file for matching images and labels.'
        self.stems = list(self.stems)

        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        image = Image.open(self.image_path.joinpath(f'{self.stems[item]}.png'))
        label = Image.open(self.label_path.joinpath(f'{self.stems[item]}.png'))

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


class MultiMaskFolder(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None):
        super(MultiMaskFolder, self).__init__()

        self.image_path = Path(root).joinpath('images/')
        self.label1_path = Path(root).joinpath('labels1/')
        self.label2_path = Path(root).joinpath('labels2/')

        self.stems = {p.stem for p in self.image_path.glob('*.png')}
        assert not self.stems ^ {p.stem for p in self.label1_path.glob('*.png')}, \
            'Missing file for matching images and labels1.'
        assert not self.stems ^ {p.stem for p in self.label2_path.glob('*.png')}, \
            'Missing file for matching images and labels2.'
        self.stems = list(self.stems)

        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, item):
        image = Image.open(self.image_path.joinpath(f'{self.stems[item]}.png'))
        label1 = Image.open(self.label1_path.joinpath(f'{self.stems[item]}.png'))
        label2 = Image.open(self.label2_path.joinpath(f'{self.stems[item]}.png'))

        if self.transform:
            image, label1, label2 = self.transform(image, label1, label2)

        return image, label1, label2
