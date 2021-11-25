'''
Author: Liu Xin
Date: 2021-11-23 18:56:12
LastEditors: Liu Xin
LastEditTime: 2021-11-23 18:57:17
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/hooks/logger/__init__.py
'''
from utils.runner.hooks.logger.base import LoggerHook
from .tensorboard import TensorboardLoggerHook
# from utils.runner.hooks.logger.text import TextLoggerHook

__all__ = ["LoggerHook", "TensorboardLoggerHook"]