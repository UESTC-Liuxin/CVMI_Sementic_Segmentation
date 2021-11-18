'''
Author: Liu Xin
Date: 2021-11-13 15:57:22
LastEditors: Liu Xin
LastEditTime: 2021-11-17 20:24:32
Description: file content
FilePath: /CVMI_Sementic_Segmentation/tools/train.py
'''

import os
import sys
from numpy.lib.type_check import imag
import yaml
import numpy as np
import argparse
from easydict import EasyDict
from torch.utils import data
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.static_common_utils import set_random_seeds
from utils.DDP import setup_distributed
from model import *


def parse_cfg(cfg_file):
    """
    @description  : 解析yaml配置文件
    @param  :
    @Returns  :
    """
    f = open(cfg_file, "r", encoding="utf-8")
    data = f.read()
    cfg_dict = EasyDict(yaml.load(data, Loader=yaml.Loader))
    return cfg_dict


def main(args):
    """
    @description  :
    @param  :
    @Returns  :
    """
    set_random_seeds()  # 设置随机种子数
    cfg_dict = parse_cfg(args.cfg_file)  # 解析config文件
    global_cfg = cfg_dict.Global
    is_distributed = global_cfg.distributed

    # 构建模型
    model_cfg = cfg_dict.Model
    print(model_cfg)
    model = build_model(model_cfg)
    if is_distributed:
        setup_distributed()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        # 1. define network
        model = model.to(device)
        # DistributedDataParallel
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    while True:
        image = torch.randn(2, 3, 256, 256).to(device)
        out = model(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation...')
    parser.add_argument('--local_rank', default=0, type=str)
    parser.add_argument(
        '-c', '--cfg_file', default="/home/liuxin/Documents/CVMI_Sementic_Segmentation/configs/base_demo.yml", type=str)
    args = parser.parse_args()
    main(args)
