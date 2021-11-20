'''
Author: Liu Xin
Date: 2021-11-18 16:25:57
LastEditors: Liu Xin
LastEditTime: 2021-11-18 16:26:12
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/visualization/summaries.py
'''
import os
import torch
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, tag,img,global_step):
        grid_image = make_grid(img,normalize=False, range=(0, 255))
        writer.add_image(tag, grid_image, global_step)