'''
Author: Liu Xin
Date: 2021-11-13 19:11:06
LastEditors: Liu Xin
LastEditTime: 2021-11-13 19:12:09
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
