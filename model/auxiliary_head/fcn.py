'''
Author: Liu Xin
Date: 2021-11-16 17:11:34
LastEditors: Liu Xin
LastEditTime: 2021-11-16 17:31:46
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/auxiliary_head/fcn.py
'''
from ..builder import AUXILIARY_HEAD
import torch.nn as nn


@AUXILIARY_HEAD.register_module("FCN")
class FCN(nn.Module):
    def __init__(self,  BatchNorm, output_stride, num_classes):

        super(FCN, self).__init__()
        self.output_stride = output_stride
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3,
                                                 stride=1, padding=1),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, inputs):
        x = inputs[1]  # 取倒数第二层
        x = self.last_conv(x)
        x = F.interpolate(x, scale_factor=self.output_stride/2,
                          mode='bilinear', align_corners=True)

        return x
