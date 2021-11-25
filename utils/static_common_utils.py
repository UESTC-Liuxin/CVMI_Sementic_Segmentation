'''
Author: Liu Xin
Date: 2021-11-13 19:11:06
LastEditors: Liu Xin
LastEditTime: 2021-11-25 15:44:12
Description: 静态工具库
FilePath: /CVMI_Sementic_Segmentation/utils/static_common_utils.py
'''
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import warnings
from socket import gethostname


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

def getuser():
    """Get the username from the environment or password database.

    First try various environment variables, then the password
    database.  This works on Windows as long as USERNAME is set.

    """

    for name in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        user = os.environ.get(name)
        if user:
            return user

    # If this fails, the exception will "explain" why
    import pwd
    return pwd.getpwuid(os.getuid())[0]

def get_host_info():
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)
    
    
def build_work_dir_suffix(global_cfg, data_cfg):
    info_dict = dict(
        bz=global_cfg.batch_size,
        gpus=global_cfg.gpus,
        optimizer_name= global_cfg.optimizer.name,
        lr = global_cfg.optimizer.lr,
        lr_sche=global_cfg.lr_config.policy,
        dataset=data_cfg.name
    )
    formated_list = [ f"{key}_{value}"  for key, value in info_dict.items()]
    return ".".join(formated_list)
        
        
        