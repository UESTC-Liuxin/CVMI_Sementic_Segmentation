'''
Author: Liu Xin
Date: 2021-11-18 09:58:40
LastEditors: Liu Xin
LastEditTime: 2021-11-26 12:13:14
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/ddp/__init__.py
'''
from utils.ddp.dist_utils import get_dist_info, setup_distributed, convert_sync_bn, mkdirs
from utils.ddp.mmdistributed_ddp import MMDistributedDataParallel