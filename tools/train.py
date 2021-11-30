'''
Author: Liu Xin
Date: 2021-11-13 15:57:22
LastEditors: Liu Xin
LastEditTime: 2021-11-29 21:40:46
Description: file content
FilePath: /CVMI_Sementic_Segmentation/tools/train.py
'''
import os
import sys
import yaml
import numpy as np
import argparse
from easydict import EasyDict
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("/home/liuxin/Documents/CVMI_Sementic_Segmentation")
from utils.static_common_utils import set_random_seeds, build_work_dir_suffix
from utils.ddp import setup_distributed, convert_sync_bn, mkdirs, MMDistributedDataParallel
from model import *
from dataloader import *
from utils.runner import build_optimizer, EpochBasedRunner, build_evaluator
from utils.logging import get_root_logger



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
        sync_BN = model_cfg.sync_BN
        if sync_BN:
            model = convert_sync_bn(num_workers, model)
        else:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    
    # 构建损失
    evaluator_cfg = cfg_dict.Evaluator
    evaluator =  build_evaluator(evaluator_cfg)
    # 构建数据集
    data_cfg = cfg_dict.Data
    trainset = build_dataset(data_cfg, split="train")
    valset = build_dataset(data_cfg, split="val")
    train_dataloader = build_dataloader(trainset, global_cfg, shuffle=True)
    val_dataloader = build_dataloader(valset,global_cfg, shuffle=False)
    # 构建优化器
    # TODO: easydict不支持未知key的pop
    optimizer_cfg = global_cfg.optimizer.copy()
    optimizer = build_optimizer(model, optimizer_cfg)
    
    # 构建logger的输出
    work_dir = global_cfg.work_dir
    work_dir = os.path.join(work_dir, build_work_dir_suffix(global_cfg, data_cfg))
    if not os.path.exists(work_dir):
        mkdirs(work_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, **global_cfg.log)
    _, file_name = os.path.split(args.cfg_file)
    shutil.copy(args.cfg_file, os.path.join(work_dir, file_name))
    
    # 构建训练器
    runner = EpochBasedRunner(
        device,
        model=model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logger,
        max_epochs=10
    )
    # 构建
    # 注册钩子函数: lr_schedule, logger
    runner.register_training_hooks(
        lr_config=global_cfg.lr_config,
        log_config=global_cfg.log_config,
        optimizer_config=global_cfg.optimizer_config,
        checkpoint_config=global_cfg.checkpoint_config
    )
    runner.register_evaluate_hook(
        {"evaluator":evaluator}
    )
    # 开始运行
    runner.run(
        data_loaders=[ train_dataloader,val_dataloader],
        workflow=global_cfg.workflow,
        max_epochs=global_cfg.max_epoch
    )
    
    
    
    
    
    


if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='Semantic Segmentation...')
    parser.add_argument('--local_rank', default=0, type=str)
    parser.add_argument(
        '-c', '--cfg_file', default="/home/liuxin/Documents/CVMI_Sementic_Segmentation/configs/base_demo.yml", type=str)
    args = parser.parse_args()
    main(args)