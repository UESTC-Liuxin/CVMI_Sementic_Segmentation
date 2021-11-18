'''
Author: Liu Xin
Date: 2021-11-15 15:40:31
LastEditors: Liu Xin
LastEditTime: 2021-11-17 20:20:19
Description: base unet decode head
FilePath: /CVMI_Sementic_Segmentation/model/decode_heads/unet.py
'''
from re import S
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import DECODE_HEAD


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@DECODE_HEAD.register_module("Unet")
class Unet(nn.Module):
    def __init__(self, in_channel, num_classes, factors,  bilinear=False):
        super(Unet, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.ups = nn.ModuleList()
        out_channel = in_channel
        for i in range(4):
            out_channel = out_channel // factors[i]
            self.ups.append(Up(in_channel, out_channel, bilinear))
            in_channel = out_channel
        self.outc = OutConv(out_channel, num_classes)

    def forward(self, features):
        [x5, x4, x3, x2, x1, x0] = features
        x = self.ups[0](x5, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        out = self.outc(x)

        return out


if __name__ == "__main__":
    image = torch.randn(4, 3, 512, 512)
    model = UNet(3, 11)
    model(image)
