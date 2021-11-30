'''
Author: Liu Xin
Date: 2021-11-25 19:23:11
LastEditors: Liu Xin
LastEditTime: 2021-11-25 22:05:46
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/necks/base_neck.py
'''
import torch.nn as nn
from model.builder import NECK

@NECK.register_module("BaseNeck")
class BaseNeck(nn.Module):
    
    def __init__(self, feature_indexes, **kwargs):
        super(BaseNeck, self).__init__()
        self.feature_indexes = feature_indexes
    
    def extract_features(self, input_features):
        return [input_features[i] for i in self.feature_indexes]
    
    def forward(self, features):
        return self.extract_features(features)