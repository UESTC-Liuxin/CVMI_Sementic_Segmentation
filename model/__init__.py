'''
Author: Liu Xin
Date: 2021-11-13 17:07:04
LastEditors: Liu Xin
LastEditTime: 2021-11-20 20:24:50
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/__init__.py
'''
from model.builder import build_model, build_criterion
from model.backbones import *
from model.necks import *
from model.decode_heads import *
from model.criterions import *

__all__ = ["build_model", "build_criterion"]
