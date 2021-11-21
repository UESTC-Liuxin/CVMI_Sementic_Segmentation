'''
Author: Liu Xin
Date: 2021-11-13 15:57:22
LastEditors: Liu Xin
LastEditTime: 2021-11-21 23:34:03
Description: file content
FilePath: /CVMI_Sementic_Segmentation/tools/train.py
'''
import os
import sys
from numpy.lib.shape_base import split
from torch import nn
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
import dataloader
sys.path.append("/home/liuxin/Documents/CVMI_Sementic_Segmentation")
from utils.static_common_utils import set_random_seeds
from utils.ddp import setup_distributed, convert_sync_bn
from model import *
from dataloader import *
from utils.runner import build_optimizer
from utils.logging import get_root_logger
from utils.runner import EpochBasedRunner



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
    model = build_model(model_cfg)
    if is_distributed:
        setup_distributed()
        num_workers = global_cfg.gpus
        # 由于是单节点多卡DDP，因此rank == local rank
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        # define network
        model = model.to(device)
        # distributedDataParallel
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        sync_BN = model_cfg.sync_BN
        if sync_BN:
            model = convert_sync_bn(num_workers, model)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    
    # 构建损失
    criterion_cfg = cfg_dict.Criterion
    criterion = build_criterion(criterion_cfg)
    # 构建数据集
    data_cfg = cfg_dict.Data
    trainset = build_dataset(data_cfg, split="train")
    valset = build_dataset(data_cfg, split="val")
    dataloaders = [
        
    ]
    # 构建优化器
    # TODO: easydict不支持未知key的pop
    optimizer_cfg = global_cfg.optimizer.copy()
    build_optimizer(model, optimizer_cfg)
    # 构建学习率策略
    
    # 构建logger的输出
    log_file = os.path.join(global_cfg.log_path, "log")
    if not os.path.exists(global_cfg.log_path):
        os.makedirs(global_cfg.log_path)
    logger = get_root_logger(log_file=log_file)
    # 构建训练器
    runner = EpochBasedRunner(
        model=model,
        optimizer=None,
        work_dir=global_cfg.log_path,
        logger=logger,
        max_epochs=10
    )
    runner.register_training_hooks(
        global_cfg.lr_config
    )
    
    
    
    
    
    
    


if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='Semantic Segmentation...')
    parser.add_argument('--local_rank', default=0, type=str)
    parser.add_argument(
        '-c', '--cfg_file', default="/home/liuxin/Documents/CVMI_Sementic_Segmentation/configs/base_demo.yml", type=str)
    args = parser.parse_args()
    main(args)
    