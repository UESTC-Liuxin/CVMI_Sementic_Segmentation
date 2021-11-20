'''
Author: Liu Xin
Date: 2021-11-16 15:29:11
LastEditors: Liu Xin
LastEditTime: 2021-11-20 20:17:05
Description: 
FilePath: /CVMI_Sementic_Segmentation/model/builder.py
'''
from model.compose import Compose
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


def build_decode_head(cfg):
    """
    @description  :
    @param  :
    @Returns  :
    """
    return build(cfg, DECODE_HEAD)


def build_model(model_cfg: dict):
    """
    @description  : 根据model_cfg构建网络结构
    @param  :
    @Returns  :
    """
    backbone_cfg = model_cfg.backbone
    neck_cfg = model_cfg.neck
    head_cfg = model_cfg.decode_head
    math_block_cfg = model_cfg.match_block
    # 构建backbone
    backbone = build_backbone(backbone_cfg)
    # 构建neck
    neck = build_neck(neck_cfg)
    # 构建decode_head
    head = build_decode_head(head_cfg)
    return Compose(math_block_cfg, backbone, neck, head)


def build_criterion(criterion_cfg: dict):
    """
    @description  : 构建损失函数
    @param  :
    @Returns  :
    """
    return build(criterion_cfg, CRITERION)
    
    
    
    
    