'''
Author: Liu Xin
Date: 2021-11-16 16:25:02
LastEditors: Liu Xin
LastEditTime: 2021-11-17 15:53:47
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/necks/identity.py
'''
import torch.nn as nn
from ..builder import NECK


@NECK.register_module("Identity")
class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input):
        return input
