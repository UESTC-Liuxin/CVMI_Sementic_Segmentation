'''
Author: Liu Xin
Date: 2021-11-13 19:11:06
LastEditors: Liu Xin
LastEditTime: 2021-11-21 17:57:30
Description: 静态工具库
FilePath: /CVMI_Sementic_Segmentation/utils/static_common_utils.py
'''

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_random_seeds():
    """
    @description  : 设置所有的随机数种子
    @param  :
    @Returns  :
    """
    seed = 6000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def is_method_overridden(method, base_class, derived_class):
    """检查基类的方法是否在派生类中被重写（copied by mmcv）

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method