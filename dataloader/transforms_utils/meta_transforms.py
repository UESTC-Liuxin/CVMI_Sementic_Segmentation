# -*- encoding: utf-8 -*-
'''
@File    :   meta_transforms.py
@Author  :   Liu Xin 
@CreateTime    :   2021/11/15 17:04:20
@Version :   1.0
@Contact :   xinliu1996@163.com
@License :   (C)Copyright 2020-2025 CVMI(UESTC)
@Desc    :   None
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
