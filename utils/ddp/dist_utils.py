'''
Author: Liu Xin
Date: 2021-11-21 21:14:29
LastEditors: Liu Xin
LastEditTime: 2021-11-21 21:14:30
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/DDP/dist_utils.py
'''

import os
import subprocess

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )



def convert_sync_bn(num_workers, model, ranks_group=None):
    """
    @description  : convert BN to ddp sync BN , 
        detail by https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html?highlight=convert_sync_batchnorm#torch.nn.SyncBatchNorm.convert_sync_batchnorm
    @param  :
    @Returns  :
    """
    if ranks_group is None:
        ranks_group =[ [i for i in range(num_workers)]]
    process_groups = [torch.distributed.new_group(pids) for pids in ranks_group]
    local_rank = dist.get_rank()
    for index, group in enumerate(ranks_group):
        if local_rank in group:
            local_group_index =  index
    process_group = process_groups[local_group_index]
    sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        sync_bn_network,
        device_ids=[local_rank],
        output_device=local_rank
        )
    return ddp_sync_bn_network
    
        

    
    
    