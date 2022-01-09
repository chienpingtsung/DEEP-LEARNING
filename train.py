import argparse
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, PILToTensor
from tqdm import tqdm

from lib.augment.common import Compose as myCompose
from lib.augment.vision import RandomCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, Dilation, \
    DivisiblePad
from lib.dataset.folder import ImageMaskFolder
from lib.loss.focalloss import FocalLoss
from lib.model.unet import UNet
from lib.util.common import get_device
from test import test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight')
    parser.add_argument('-l', '--logdir')
    parser.add_argument('-t', '--threshold', default=0.5, type=float)
    parser.add_argument('-b', '--batchsize', default=8, type=int)
    parser.add_argument('-d', '--dilation', default=3, type=int)
    parser.add_argument('-s', '--scout', default=10, type=int)
    parser.add_argument('--trainset', required=True)
    parser.add_argument('--testset', required=True)
    args = parser.parse_args()

    device = get_device()

    args.batchsize *= torch.cuda.device_count() if torch.cuda.is_available() else 1

    writer = SummaryWriter(args.logdir)

    trainset = ImageMaskFolder(args.trainset,
                               transforms=myCompose([RandomCrop((512, 512)),
                                                     RandomRotation(range(0, 360, 90)),
                                                     RandomHorizontalFlip(),
                                                     RandomVerticalFlip()]),
                               transform=Compose([ToTensor()]),
                               target_transform=Compose([Dilation(args.dilation),
                                                         ToTensor()]))
    testset = ImageMaskFolder(args.testset,
                              transform=Compose([DivisiblePad(),
                                                 ToTensor()]),
                              target_transform=Compose([PILToTensor()]))
    trainloader = DataLoader(trainset,
                             batch_size=args.batchsize,
                             shuffle=True,
                             num_workers=args.batchsize,
                             pin_memory=True,
                             drop_last=True)
    testloader = DataLoader(testset, pin_memory=True)

    snapshot = torch.load(args.weight) if args.weight else None

    model = UNet(3, 1)
    if snapshot:
        model.load_state_dict(snapshot['model'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters())
    if snapshot:
        optimizer.load_state_dict(snapshot['optim'])

    best_f1 = snapshot['best_f1'] if snapshot else 0
    best_f1_epoch = snapshot['best_f1_epoch'] if snapshot else 0

    for epoch in count(snapshot['epoch'] + 1 if snapshot else 0):
        model.train()
        tq = tqdm(trainloader)
        for image, mask, *_ in tq:
            output = model(image.to(device))
            loss = criterion(output, mask.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tq.set_description(f'Training epoch {epoch}, loss {loss.item()}')
            writer.add_scalar('train/loss', loss.item(), epoch)

        prec, reca, f1 = test(model, tqdm(testloader), device, threshold=args.threshold,
                              save_to=Path(writer.log_dir).joinpath(f'test/{epoch}/'))
        writer.add_scalar('test/Precision', prec, epoch)
        writer.add_scalar('test/Recall', reca, epoch)
        writer.add_scalar('test/F1', f1, epoch)

        snapshot = {'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'best_f1': f1 if f1 > best_f1 else best_f1,
                    'best_f1_epoch': epoch if f1 > best_f1 else best_f1_epoch,
                    'epoch': epoch}
        torch.save(snapshot, Path(writer.log_dir).joinpath('last.pth'))

        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch
            torch.save(snapshot, Path(writer.log_dir).joinpath('best.pth'))

        if epoch - best_f1_epoch > args.scout:
            writer.add_text('U-Net', f'Best model at epoch {best_f1_epoch}, and F1 {best_f1}.')
            break
