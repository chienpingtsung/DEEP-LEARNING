import torch
from torch import nn

from models.seg1 import Bottleneck


def double_conv(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Seg2(nn.Module):
    """
    https://arxiv.org/pdf/1505.04597
    """

    def __init__(self, in_channels, out_channels):
        super(Seg2, self).__init__()

        # contracting path
        self.cont1 = double_conv(in_channels, 64)
        self.maxpool1 = nn.MaxPool2d(2)
        self.cont2 = double_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(2)
        self.cont3 = double_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(2)
        self.cont4 = double_conv(256, 512)
        self.maxpool4 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = double_conv(512, 1024)

        # seg expansive path
        self.seg_up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.seg_expa1 = double_conv(1024, 512)
        self.seg_up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_expa2 = double_conv(512, 256)
        self.seg_up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_expa3 = double_conv(256, 128)
        self.seg_up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_expa4 = double_conv(128, 64)

        # edge expansive path
        self.edge_up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.edge_expa1 = double_conv(1024, 512)
        self.edge_up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.edge_expa2 = double_conv(512, 256)
        self.edge_up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.edge_expa3 = double_conv(256, 128)
        self.edge_up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.edge_expa4 = double_conv(128, 64)

        # merge
        self.merge_bottleneck = Bottleneck(128, 128 // Bottleneck.expansion, groups=32, base_width=8)

        # output
        self.seg_output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.edge_output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.merge_output = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # contracting path
        x1 = self.cont1(x)
        x = self.maxpool1(x1)
        x2 = self.cont2(x)
        x = self.maxpool2(x2)
        x3 = self.cont3(x)
        x = self.maxpool3(x3)
        x4 = self.cont4(x)
        x = self.maxpool4(x4)

        # bottleneck
        x5 = self.bottleneck(x)

        # seg expansive path
        x = self.seg_up1(x5)
        x = self.seg_expa1(torch.cat((x4, x), dim=1))
        x = self.seg_up2(x)
        x = self.seg_expa2(torch.cat((x3, x), dim=1))
        x = self.seg_up3(x)
        x = self.seg_expa3(torch.cat((x2, x), dim=1))
        x = self.seg_up4(x)
        x = self.seg_expa4(torch.cat((x1, x), dim=1))

        # edge expansive path
        y = self.edge_up1(x5)
        y = self.edge_expa1(torch.cat((x4, y), dim=1))
        y = self.edge_up2(y)
        y = self.edge_expa2(torch.cat((x3, y), dim=1))
        y = self.edge_up3(y)
        y = self.edge_expa3(torch.cat((x2, y), dim=1))
        y = self.edge_up4(y)
        y = self.edge_expa4(torch.cat((x1, y), dim=1))

        # merge
        z = self.merge_bottleneck(torch.cat((x, y), dim=1))

        # output
        return self.seg_output(x), self.edge_output(y), self.merge_output(z)
