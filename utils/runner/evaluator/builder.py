'''
Author: Liu Xin
Date: 2021-11-23 10:30:18
LastEditors: Liu Xin
LastEditTime: 2021-11-23 15:48:17
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/evaluator/builder.py
'''

from utils.registry import Registry, build

EVALUATOR = Registry("evaluator")

def build_evaluator(cfg):
    print(EVALUATOR)
    return build(cfg, EVALUATOR)

