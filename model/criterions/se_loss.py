'''
Author: Liu Xin
Date: 2021-11-18 09:58:40
LastEditors: Liu Xin
LastEditTime: 2021-11-29 22:15:03
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/criterions/SELoss.py
'''
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.builder import CRITERION

@CRITERION.register_module()
class SELoss(nn.Module):
    def __init__(self, num_classes=6, *args, **kwargs):
        super(SELoss, self).__init__()
        self.bce = nn.BCELoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        target = self._get_batch_label_vector(target, self.num_classes)
        loss = self.bce(torch.sigmoid(pred), target.to(pred.device))
        return loss

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                            bins=nclass, min=0,
                            max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

    

if __name__ == "__main__":
    pred = torch.randn(2, 6)
    gt = torch.randn(2, 6).random_(6).long()
    lossfunc = SELoss(num_classes=6)
    loss = lossfunc(pred, gt)
    print(loss)
