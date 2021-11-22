'''
Author: Liu Xin
Date: 2021-11-19 21:18:50
LastEditors: Liu Xin
LastEditTime: 2021-11-22 10:00:13
Description: file content
FilePath: /CVMI_Sementic_Segmentation/dataloader/builder.py
'''
from random import shuffle
from torch import distributed
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler,Sampler, RandomSampler
from utils.registry import Registry,build
from torchvision import transforms

DATASET = Registry("dataset")
TRANSFORMS = Registry("transforms")


def build_dataset(data_cfg, split="train"):
    """
    @description  :
    @param  :
    @Returns  :
    """
    transforms_cfg = data_cfg.transforms[split]
    transforms = build_transforms(transforms_cfg)
    data_cfg = data_cfg.copy()
    data_cfg["transforms"] = transforms
    data_cfg["split"] = split
    return build(data_cfg, DATASET)
    
def build_transforms(transforms_cfg):
    """
    @description  :构建
    @param  :
    @Returns  :
    """
    def build(name, cfg, registry: Registry):
        obj_cls = registry.module_dict[name]
        if isinstance(cfg, dict):
            return obj_cls(**cfg)
        if isinstance(cfg, list):
            return obj_cls(*cfg)
        if cfg is None:
            return obj_cls()
    transforms_list = []
    for key, value in transforms_cfg.items():
        transforms_list.append(build(key, value, TRANSFORMS))
    return transforms.Compose(transforms_list)



def build_dataloader(dataset, cfg, shuffle=True):
    """
    @description  :
    @param  :
    @Returns  :
    """
    distributed = cfg.distributed
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    if distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            shuffle = shuffle
        )
    else:
        sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
        
    

    