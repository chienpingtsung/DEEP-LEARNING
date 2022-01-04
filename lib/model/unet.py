import torch
from torch import nn


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


class UNet(nn.Module):
    """
    https://arxiv.org/pdf/1505.04597
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

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

        # expansive path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.expa1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.expa2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.expa3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.expa4 = double_conv(128, 64)

        # output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

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
        x = self.bottleneck(x)

        # expansive path
        x = self.up1(x)
        x = self.expa1(torch.cat((x4, x), dim=1))
        x = self.up2(x)
        x = self.expa2(torch.cat((x3, x), dim=1))
        x = self.up3(x)
        x = self.expa3(torch.cat((x2, x), dim=1))
        x = self.up4(x)
        x = self.expa4(torch.cat((x1, x), dim=1))

        # output
        return self.output(x)
