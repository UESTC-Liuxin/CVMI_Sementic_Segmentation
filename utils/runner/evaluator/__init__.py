'''
Author: Liu Xin
Date: 2021-11-23 10:30:08
LastEditors: Liu Xin
LastEditTime: 2021-11-23 15:47:14
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/evaluator/__init__.py
'''
from utils.runner.evaluator.builder import build_evaluator
from utils.runner.evaluator.evaluator import BaseEvaluator

__all__ = ["build_evaluator","BaseEvaluator"]