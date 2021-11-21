'''
Author: Liu Xin
Date: 2021-11-21 18:10:58
LastEditors: Liu Xin
LastEditTime: 2021-11-21 21:38:30
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/optimizer/builder.py
'''
import copy
import inspect
import torch
from utils.registry import Registry, build

OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')


def register_torch_optimizers():
    """
    @description  : 通过扫描torvh.optim文件，获取torch实现的优化器对象
    @param  :
    @Returns  :
    """
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(_optim.__name__)(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers

TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg):
    return build(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            name=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer