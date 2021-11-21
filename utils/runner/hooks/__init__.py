'''
Author: Liu Xin
Date: 2021-11-21 17:47:23
LastEditors: Liu Xin
LastEditTime: 2021-11-21 23:25:35
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/hooks/__init__.py
'''
from utils.runner.hooks.hook import Hook,HOOKS
from utils.runner.hooks.lr_updater import LrUpdaterHook
__all__ = [
    'HOOKS', 'Hook', 'LrUpdaterHook'
]
