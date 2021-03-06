from typing import Optional, Callable, Type, Union, List

import torch
from torch import nn, Tensor

from models.unet import double_conv


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html"""
    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Seg1(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block: Optional[Type[Union[Bottleneck]]] = None,
                 layers: Optional[List[int]] = None,
                 zero_init_residual: bool = False,
                 groups: int = 32,
                 width_per_group: int = 8,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Seg1, self).__init__()
        if block is None:
            block = Bottleneck
        if layers is None:
            layers = [3, 4, 23, 3]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        # stage conv1
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # stage conv2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # stage conv3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # stage conv4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # stage conv5
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # expansive path
        self.up1 = nn.ConvTranspose2d(512 * block.expansion, 256 * block.expansion, kernel_size=2, stride=2)
        self.expa1 = double_conv(512 * block.expansion, 256 * block.expansion)
        self.up2 = nn.ConvTranspose2d(256 * block.expansion, 128 * block.expansion, kernel_size=2, stride=2)
        self.expa2 = double_conv(256 * block.expansion, 128 * block.expansion)
        self.up3 = nn.ConvTranspose2d(128 * block.expansion, 64 * block.expansion, kernel_size=2, stride=2)
        self.expa3 = double_conv(128 * block.expansion, 64 * block.expansion)
        self.up4 = nn.ConvTranspose2d(64 * block.expansion, 32 * block.expansion, kernel_size=2, stride=2)
        self.expa4 = double_conv(32 * block.expansion + 64, 32 * block.expansion)
        # output
        self.output = nn.Conv2d(32 * block.expansion, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: Type[Union[Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # stage conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # stage conv2
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        # stage conv3
        x3 = self.layer2(x2)
        # stage conv4
        x4 = self.layer3(x3)
        # stage conv5
        x = self.layer4(x4)
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

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
