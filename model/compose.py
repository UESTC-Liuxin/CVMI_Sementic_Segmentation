'''
Author: Liu Xin
Date: 2021-11-16 16:51:42
LastEditors: Liu Xin
LastEditTime: 2021-11-21 22:54:21
Description: compose all sub-model to a segmentation pipeline
FilePath: /CVMI_Sementic_Segmentation/model/compose.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchBlock(nn.Module):
    def __init__(self, image_size, type="bilinear") -> None:
        super(MatchBlock, self).__init__()
        self.match_block = nn.Upsample(size=image_size, mode=type)

    def forward(self, input):
        out = self.match_block(input)
        return out


class Compose(nn.Module):
    def __init__(self, match_cfg,  backbone, neck, decode_head, auxiliary=None) -> None:
        super(Compose, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.auxiliary = auxiliary
        self.match_block = MatchBlock(**match_cfg)
        if (auxiliary is not None):
            self.match_block_auxiliary = MatchBlock(**match_cfg)

    def forward(self, input):
        outs = dict()
        features = self.backbone(input)
        features = self.neck(features)
        if self.auxiliary is not None:
            auxilary_base_out = self.auxiliary(features)
            auxilary_out = self.match_block_auxiliary(auxilary_base_out)
            outs["auxiliary_out"] = auxilary_out
        base_out = self.decode_head(features)
        out = self.match_block(base_out)
        outs["out"] = out
        return outs
    
    def train_step(self, data_batch, **kwargs):
        """
        @description  :
        @param  :
        @Returns  :
        """
        outs  = self(**data_batch)
        return outs
    
    def val_step(self, data_batch, **kwargs):
        """
        @description  :
        @param  :
        @Returns  :
        """
        outs  = self(**data_batch)
        return outs
    
        
        
        
    
    
