'''
Author: Liu Xin
Date: 2021-11-21 21:14:29
LastEditors: Liu Xin
LastEditTime: 2021-11-26 12:37:59
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/ddp/dist_utils.py
'''

import os
import subprocess
import pickle
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
            os.environ["MASTER_PORT"] = "29501"
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
        output_device=local_rank,
        find_unused_parameters=True
        )
    return ddp_sync_bn_network
    
        
def collect_results_gpu(result_part, size=None):
    """Collect results under gpu mode.
        
    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.
    在分布式模式下对数据进行汇总（在val和test过程中需要同步评估结论）
    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    if not dist.is_initialized():
        return
    if size is None:
        size = len(result_part)
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    
@master_only
def mkdirs(*args, **kwargs):
    os.makedirs(*args, **kwargs)