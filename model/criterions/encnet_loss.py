'''
Author: Liu Xin
Date: 2021-11-29 22:00:14
LastEditors: Liu Xin
LastEditTime: 2021-11-29 22:54:32
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/criterions/encnet_loss.py
'''
import torch
import torch.nn as nn

from model.criterions import ImageBasedCrossEntropy2D
from model.criterions import SELoss
from model.builder import CRITERION

@CRITERION.register_module("encnet_loss")
class EncNetLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EncNetLoss, self).__init__()
        self.cross_entropy2d = ImageBasedCrossEntropy2D(*args, **kwargs)
        self.se = SELoss(*args, **kwargs)
        
    def forward(self, pred, se_out, mask) :
        seg_loss = self.cross_entropy2d(pred, mask)
        se_loss = self.se(se_out, mask)
        return seg_loss, se_loss
         