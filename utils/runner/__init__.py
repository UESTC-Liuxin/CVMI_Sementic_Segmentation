'''
Author: Liu Xin
Date: 2021-11-21 20:53:33
LastEditors: Liu Xin
LastEditTime: 2021-11-25 10:08:53
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/__init__.py
'''

from utils.runner.optimizer.builder import build_optimizer
from utils.runner.epoch_base_runner import EpochBasedRunner
from utils.runner.evaluator import build_evaluator
from utils.runner.checkpoint import *