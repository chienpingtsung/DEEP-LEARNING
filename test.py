import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.folder import MaskFolder
from models.unet import UNet
from transforms.translate import ToTensor
from transforms.utils import Compose
from transforms.vision import Resize
from utils.evaluate import calc_confusion_matrix


def test(model, dataloader, device, threshold=0.5, save_to: Path = None):
    if save_to:
        save_to.mkdir(parents=True, exist_ok=True)

    model.eval()

    prec_TP = prec_TPFP = reca_TP = reca_TPFN = 0

    for image, label in dataloader:
        with torch.no_grad():
            image = image.to(device)

            output = model(image)
            output = torch.sigmoid(output)
            pred = torch.squeeze(output) > threshold

            label = torch.squeeze(label).numpy().astype(bool)
            pred = pred.cpu().numpy()
            for la, pr in zip(label, pred):
                p_TP, p_TPFP, r_TP, r_TPFN = calc_confusion_matrix(la, pr, tolerance=2)
                prec_TP += p_TP
                prec_TPFP += p_TPFP
                reca_TP += r_TP
                reca_TPFN += r_TPFN

    infinitesimal = 1e-10
    prec = prec_TP / (prec_TPFP + infinitesimal)
    reca = reca_TP / (reca_TPFN + infinitesimal)
    f1 = (2 * prec * reca) / (prec + reca + infinitesimal)

    return prec, reca, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{torch.cuda.device_count()} cuda device available.')
    print(f'Using {device} device.')

    batch_size = args.batch_size
    if torch.cuda.device_count() > 1:
        batch_size *= torch.cuda.device_count()

    testset = MaskFolder(args.dataset,
                         transform=Compose([
                             Resize((512, 512)),
                             ToTensor()
                         ]))
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=batch_size,
                            pin_memory=True,
                            drop_last=False)

    model = UNet(3, 1)
    model.load_state_dict(torch.load(args.weights))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    prec, reca, f1 = test(model,
                          tqdm(testloader, desc=f'Testing threshold {args.threshold}'),
                          device,
                          threshold=args.threshold)
    print(f'Precision {prec}. Recall {reca}. F1 {f1}.')
