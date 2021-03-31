import argparse
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.folder import MultiMaskFolder
from losses.focalloss import FocalLoss
from models.seg2 import Seg2
from testseg2 import test
from transforms.translate import ToTensor
from transforms.utils import Compose
from transforms.vision import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, Dilation, RandomCrop

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--log_dir')
parser.add_argument('--weights')
parser.add_argument('--train_backbone', default=True, type=bool)
parser.add_argument('--train_seg', default=True, type=bool)
parser.add_argument('--train_edge', default=True, type=bool)
parser.add_argument('--train_merge', default=True, type=bool)
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

trainset = MultiMaskFolder('/home/chienping/JupyterLab/datasets/07v2crack/train/',
                           transform=Compose([RandomHorizontalFlip(),
                                              RandomVerticalFlip(),
                                              RandomRotation(0, 360, expand=True),
                                              RandomCrop((512, 512)),
                                              ToTensor()]),
                           label1_transform=Dilation(3),
                           label2_transform=Dilation(3))
trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=batch_size,
                         pin_memory=True,
                         drop_last=True)

model = Seg2(1, 1)
if args.weights:
    model.load_state_dict(torch.load(args.weights))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = FocalLoss()
for p in model.parameters():
    p.requires_grad = False
if args.train_backbone:
    for k, v in model.named_parameters():
        if k.startswith(('cont', 'maxpool', 'bottleneck')):
            v.requires_grad = True
if args.train_seg:
    for k, v in model.named_parameters():
        if k.startswith('seg_'):
            v.requires_grad = True
if args.train_edge:
    for k, v in model.named_parameters():
        if k.startswith('edge_'):
            v.requires_grad = True
if args.train_merge:
    for k, v in model.named_parameters():
        if k.startswith('merge_'):
            v.requires_grad = True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

best_f1 = 0
best_f1_epoch = 0

for epoch in count(args.start_epoch):
    model.train()
    total_loss = 0
    propagation_counter = 0
    tq = tqdm(trainloader)
    for image, label1, label2 in tq:
        image = image.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)

        seg_output, edge_output, merge_output = model(image)
        seg_loss = criterion(seg_output, label1)
        edge_loss = criterion(edge_output, label2)
        merge_loss = criterion(merge_output, label1)

        loss = 0
        if args.train_seg:
            loss += seg_loss
        if args.train_edge:
            loss += edge_loss
        if args.train_merge:
            loss += merge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        propagation_counter += 1
        tq.set_description(f'Training epoch {epoch}, loss {loss.item()}')
        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/seg_loss', seg_loss.item(), epoch)
        writer.add_scalar('train/edge_loss', edge_loss.item(), epoch)
        writer.add_scalar('train/merge_loss', merge_loss.item(), epoch)

    scheduler.step(total_loss / propagation_counter)
    writer.add_scalar('train/mean_loss', total_loss / propagation_counter, epoch)

    seg_prec, seg_reca, seg_f1, \
    edge_prec, edge_reca, edge_f1, \
    merge_prec, merge_reca, merge_f1 \
        = test(model,
               image_root=Path('/home/chienping/JupyterLab/datasets/07v2crack/val/images/'),
               label1_root=Path('/home/chienping/JupyterLab/datasets/07v2crack/val/labels1/'),
               label2_root=Path('/home/chienping/JupyterLab/datasets/07v2crack/val/labels2/'),
               batch_size=batch_size * 2,
               device=device)

    writer.add_scalar('test/seg_Precision', seg_prec, epoch)
    writer.add_scalar('test/seg_Recall', seg_reca, epoch)
    writer.add_scalar('test/seg_F1', seg_f1, epoch)
    writer.add_scalar('test/edge_Precision', edge_prec, epoch)
    writer.add_scalar('test/edge_Recall', edge_reca, epoch)
    writer.add_scalar('test/edge_F1', edge_f1, epoch)
    writer.add_scalar('test/merge_Precision', merge_prec, epoch)
    writer.add_scalar('test/merge_Recall', merge_reca, epoch)
    writer.add_scalar('test/merge_F1', merge_f1, epoch)
    print(f'Epoch {epoch}. Precision {merge_prec}. Recall {merge_reca}. F1 {merge_f1}.')

    if isinstance(model, DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, Path(writer.log_dir).joinpath('last.pth'))

    eval_f1 = seg_f1
    if args.train_edge:
        eval_f1 = edge_f1
    if args.train_merge:
        eval_f1 = merge_f1
    if eval_f1 > best_f1:
        best_f1 = eval_f1
        best_f1_epoch = epoch
        torch.save(state_dict, Path(writer.log_dir).joinpath('best.pth'))

    if epoch - best_f1_epoch > 20:
        writer.add_text('U-Net', f'Best model at epoch {best_f1_epoch}, and F1 {best_f1}.')
        break
