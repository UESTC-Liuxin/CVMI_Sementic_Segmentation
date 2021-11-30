'''
Author: Liu Xin
Date: 2021-11-30 19:25:32
LastEditors: Liu Xin
LastEditTime: 2021-11-30 20:09:13
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/decode_heads/unet/enc_unet.py
'''
from re import S
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.builder import DECODE_HEAD
from model.decode_heads.unet import Unet
from model.utils.enc_module import EncModule

@DECODE_HEAD.register_module("enc_unet")
class EncUnet(Unet):
    def __init__(self, in_channels, num_classes, factors, criterion, match_block, bilinear=False,*args, **kwargs):
        super().__init__(in_channels, num_classes, factors, criterion, match_block, bilinear=bilinear)
        self.encmodule = EncModule(512, num_classes, ncodes=32, se_loss=se_loss,
                            norm_layer=nn.BatchNorm2d,**kwargs)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1, False),
            nn.Conv2d(512, num_classes, 1)
        )
        
    def  forward(self, features, data_batch):
        [x0, x1, x2, x3, x4] = features
        x = self.ups[0](x4, x3)
        x = self.ups[1](x, x2)
        x = self.ups[2](x, x1)
        x = self.ups[3](x, x0)
        
        seg_out, se_out = list(self.encmodule(x))
        base_out = self.outc(seg_out)
        out = self.match_block(base_out)
        seg_loss, se_loss = self.criterion(out, se_out, data_batch["mask"])
        return {"seg_out":out, " seg_loss":seg_loss, "se_loss":se_loss}
    