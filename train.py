import argparse
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.folder import MaskFolder
from losses.focalloss import FocalLoss
from models.seg1 import Seg1
from test import test
from transforms.translate import ToTensor
from transforms.utils import Compose
from transforms.vision import Resize

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--log_dir')
parser.add_argument('--weights')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--start_epoch', default=0, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{torch.cuda.device_count()} cuda device available.')
print(f'Using {device} device.')

batch_size = args.batch_size
if torch.cuda.device_count() > 1:
    batch_size *= torch.cuda.device_count()
writer = SummaryWriter(args.log_dir)

trainset = MaskFolder('/home/chienping/JupyterLab/datasets/04v2crack/train/',
                      transform=Compose([
                          Resize((512, 512)),
                          ToTensor()
                      ]))
testset = MaskFolder('/home/chienping/JupyterLab/datasets/04v2crack/val/',
                     transform=Compose([
                         Resize((512, 512)),
                         ToTensor()
                     ]))
trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=batch_size,
                         pin_memory=True,
                         drop_last=True)
testloader = DataLoader(testset,
                        batch_size=batch_size * 2,
                        shuffle=False,
                        num_workers=batch_size,
                        pin_memory=True,
                        drop_last=False)

model = Seg1(3, 1, width_per_group=4, layers=[3, 4, 6, 3])
if args.weights:
    model.load_state_dict(torch.load(args.weights))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

best_f1 = 0
best_f1_epoch = 0

for epoch in count(args.start_epoch):
    model.train()
    total_loss = 0
    propagation_counter = 0
    tq = tqdm(trainloader)
    for image, label in tq:
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        propagation_counter += 1
        tq.set_description(f'Training epoch {epoch}, loss {loss.item()}')
        writer.add_scalar('train/loss', loss.item(), epoch)

    scheduler.step(total_loss / propagation_counter)
    writer.add_scalar('train/mean_loss', total_loss / propagation_counter, epoch)

    prec, reca, f1 = test(model,
                          tqdm(testloader, desc=f'Testing epoch {epoch}'),
                          device,
                          Path(writer.log_dir).joinpath(f'test/{epoch}/'))

    writer.add_scalar('test/Precision', prec, epoch)
    writer.add_scalar('test/Recall', reca, epoch)
    writer.add_scalar('test/F1', f1, epoch)
    print(f'Epoch {epoch}. Precision {prec}. Recall {reca}. F1 {f1}.')

    if isinstance(model, DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, Path(writer.log_dir).joinpath('last.pth'))

    if f1 > best_f1:
        best_f1 = f1
        best_f1_epoch = epoch
        torch.save(state_dict, Path(writer.log_dir).joinpath('best.pth'))

    if epoch - best_f1_epoch > 10:
        writer.add_text('U-Net', f'Best model at epoch {best_f1_epoch}, and F1 {best_f1}.')
        break
