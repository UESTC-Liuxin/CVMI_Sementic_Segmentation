'''
Author: Liu Xin
Date: 2021-11-16 16:51:42
LastEditors: Liu Xin
LastEditTime: 2021-11-22 20:34:16
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
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, match_cfg,  backbone, neck, decode_heads:nn.ModuleDict):
        super(Compose, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_heads = decode_heads
        self.match_block = nn.ModuleDict()
        # 由于不知道会有多少个head
        for key, head in self.decode_heads.items():
            if "seg" in key:
                self.match_block[key] = MatchBlock(**match_cfg)
        
    def forward_train(self, image,**kwargs):
        """
        @description  : 训练时前向推断，经过auxiliary
        @param  :
        @Returns  :
        """
        outs = kwargs
        features = self.backbone(image )
        features = self.neck(features)
        for key, head in self.decode_heads.items():
            out = head(features)
            # 只有*SegHead才需要进行尺寸匹配
            if "seg" in key:
                out = self.match_block[key] (out)
            outs[key+"_out"] =out
        return outs
    
    def forward_val(self, image, **kwargs):
        """
        @description  : 评估的前向推断(原则上是不经过auxiliary的）
        @param  :
        @Returns  :
        """
        outs = dict()
        features = self.backbone(image )
        features = self.neck(features)
        for key, head in self.decode_heads.items():
            out = head(features)
            # 只有*SegHead才需要进行尺寸匹配
            if "Seg" in key:
                out = self.match_block[key] (out)
            outs[key+"_out"] =out
        return outs

    
    def train_step(self, data_batch, optimizer,  **kwargs):
        """
        @description  :
        @param  :
        @Returns  :
        """
        outs  = self.forward_train(**data_batch)
        return outs
    
    def val_step(self, data_batch, optimizer, **kwargs):
        """
        @description  :
        @param  :
        @Returns  :
        """
        outs  = self.forward_val(**data_batch)
        return outs
    
        
        
        
    
    
