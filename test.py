import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from lib.augment.vision import DivisiblePad
from lib.dataset.folder import ImageMaskFolder
from lib.model.unet import UNet
from lib.util.common import get_device
from lib.util.evaluate import calc_confusion_matrix


def test(model, dataloader, device, threshold=0.5, save_to: str = None):
    if save_to:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)

    model.eval()

    prec_TP = prec_TPFP = reca_TP = reca_TPFN = 0

    for image, mask, stem, size in dataloader:
        with torch.no_grad():
            output = model(image.to(device))
            output = torch.sigmoid(output)
            pred = torch.squeeze(output) > threshold
            pred = pred.cpu().numpy()

            p_TP, p_TPFP, r_TP, r_TPFN = calc_confusion_matrix(mask, pred, tolerance=2)
            prec_TP += p_TP
            prec_TPFP += p_TPFP
            reca_TP += r_TP
            reca_TPFN += r_TPFN

            if save_to:
                pred = pred.astype(np.uint8) * 255
                Image.fromarray(pred, 'L').convert('1').save(save_to.joinpath(f'{stem}.png'))

    infinitesimal = 1e-10
    prec = prec_TP / (prec_TPFP + infinitesimal)
    reca = reca_TP / (reca_TPFN + infinitesimal)
    f1 = (2 * prec * reca) / (prec + reca + infinitesimal)

    return prec, reca, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True)
    parser.add_argument('-d', '--testset', required=True)
    parser.add_argument('-t', '--threshold', default=0.5, type=float)
    parser.add_argument('-s', '--save')
    args = parser.parse_args()

    device = get_device()

    testset = ImageMaskFolder(args.testset,
                              transform=Compose([DivisiblePad(),
                                                 ToTensor()]),
                              target_transform=Compose([np.asarray]))
    testloader = DataLoader(testset, pin_memory=True)

    model = UNet(3, 1)
    model.load_state_dict(torch.load(args.weight)['model'])
    model.to(device)

    prec, reca, f1 = test(model, tqdm(testloader), device, threshold=args.threshold, save_to=args.save)
    print(f'Precision {prec}. Recall {reca}. F1 {f1}.')
