'''
Author: Liu Xin
Date: 2021-11-13 15:57:22
LastEditors: Liu Xin
LastEditTime: 2021-11-15 20:42:19
Description: file content
FilePath: /CVMI_Sementic_Segmentation/tools/train.py
'''
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Author  :   Liu Xin 
@CreateTime    :   2021/11/13 16:55:59
@Version :   1.0
@Contact :   xinliu1996@163.com
@License :   (C)Copyright 2020-2025 CVMI(UESTC)
@Desc    :   None
'''




import sys
import yaml
import numpy as np
import argparse
from torch.utils import data
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.static_common_utils import set_random_seeds
def parse_cfg(cfg_file):
    """
    @description  : 解析yaml配置文件
    @param  :
    @Returns  :
    """
    f = open(cfg_file, "r", encoding="utf-8")
    data = f.read()
    cfg_dict = yaml.load(data, Loader=yaml.Loader)
    return cfg_dict


def main(args):
    """
    @description  :
    @param  :
    @Returns  :
    """
    cfg_dict = parse_cfg(args.cfg_file)  # 解析config文件
    set_random_seeds()  # 设置随机种子数
    # 构建模型
    print(cfg_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation...')
    parser.add_argument('--local_rank', default=0, type=str)
    parser.add_argument('-c', '--cfg_file', type=str)
    args = parser.parse_args()
    main(args)
