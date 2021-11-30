'''
Author: Liu Xin
Date: 2021-11-16 15:29:11
LastEditors: Liu Xin
LastEditTime: 2021-11-29 21:42:02
Description: 
FilePath: /CVMI_Sementic_Segmentation/model/builder.py
'''
import torch.nn as nn
import copy
from model.compose import Compose, MatchBlock
from utils.registry import Registry, build


BACKBONE = Registry("backbone")
NECK = Registry("neck")
DECODE_HEAD = Registry("decode_head")
AUXILIARY_HEAD = Registry("auxiliary_head")
CRITERION = Registry("criterion")



def build_backbone(cfg):
    """
    @description  :对backbone进行构建
    @param  :
    @Returns  :
    """
    return build(cfg, BACKBONE)


def build_neck(cfg):
    """
    @description  : 对neck进行构建
    @param  :
    @Returns  :
    """
    return build(cfg, NECK)


def build_decode_heads(cfg):
    """
    @description  : 由于decode_head 可能会有多个，因此利用nn.ModuleDict()来进行储存
    @param  :
    @Returns  :
    """
    def build_decode_head(head_cfg):
        match_block_cfg = head_cfg.pop("match_block")
        criterion_cfg = head_cfg.pop("criterion")
        match_block = MatchBlock(**match_block_cfg)
        criterion = build_criterion(criterion_cfg)
        return build(
            dict(criterion=criterion, match_block= match_block, **head_cfg), 
            DECODE_HEAD
        )
        
    decode_heads = nn.ModuleDict()
    for key, decode_head_cfg in cfg.items():
        assert isinstance(decode_head_cfg, dict)
        decode_heads[key] = build_decode_head(decode_head_cfg)
    return decode_heads

def build_model(model_cfg: dict):
    """
    @description  : 根据model_cfg构建网络结构
    @param  :
    @Returns  :
    """
    backbone_cfg = model_cfg.backbone
    neck_cfg = model_cfg.neck
    head_cfg = model_cfg.decode_heads
    # 构建backbone
    backbone = build_backbone(backbone_cfg)
    # 构建neck
    neck = build_neck(neck_cfg)
    # 构建decode_head
    decode_heads = build_decode_heads(head_cfg)
    return Compose(backbone, neck, decode_heads)


def build_criterion(criterion_cfg: dict):
    """
    @description  : 构建损失函数字典
    @param  : decode_heads_cfg
    @Returns  :
    """
    return build(criterion_cfg, CRITERION)
     
    
    
    
    
    