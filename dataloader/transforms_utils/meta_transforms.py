'''
Author: Liu Xin
Date: 2021-11-18 09:58:40
LastEditors: Liu Xin
LastEditTime: 2021-11-19 19:38:07
Description: file content
FilePath: /CVMI_Sementic_Segmentation/dataloader/transforms_utils/meta_transforms.py
'''
import torch


class MetaToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['section'] = torch.Tensor(sample)
        return sample
