'''
Author: Liu Xin
Date: 2021-11-16 16:51:42
LastEditors: Liu Xin
LastEditTime: 2021-11-29 21:44:45
Description: compose all sub-model to a segmentation pipeline
FilePath: /CVMI_Sementic_Segmentation/model/compose.py
'''
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ddp.dist_utils import dist

class MatchBlock(nn.Module):
    def __init__(self, image_size, type="bilinear") -> None:
        super(MatchBlock, self).__init__()
        self.match_block = nn.Upsample(size=image_size, mode=type, align_corners=False)

    def forward(self, input):
        out = self.match_block(input)
        return out


class Compose(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
        attention_hooks: 将特定的中间输出取出至特定的层
    """
    def __init__(self, backbone, neck, decode_heads:nn.ModuleDict):
        super(Compose, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_heads = decode_heads
        del self.backbone.fc
        
    def forward(self, data_batch, optimizer, train_mode,  **kwargs):
        if train_mode:
            outputs  = self.forward_train(**data_batch)
        else:
            outputs  = self.forward_val(**data_batch)

        return outputs
    
    def forward_train(self, **data_batch):
        """
        @description  : 训练时前向推断，经过auxiliary
        @param  :
        @Returns  :
        """
        losses = dict()
        head_outs = dict()
        features = self.backbone(data_batch["image"] )
        features = self.neck(features)
        for head_key, head in self.decode_heads.items():
            results = head(features, data_batch)
            for key, value in results.items():
                if "out" in key:
                    head_outs[head_key + "_" + key] = value
                if "loss" in key:
                    losses[head_key + "_" + key] = value
                    if "loss" not in losses:
                        losses.setdefault("loss", value)
                    else:
                        losses["loss"] += value
                        
        log_vars = self._parse_losses(losses)
        outputs = dict(
                log_vars=log_vars,
                num_samples=len(data_batch['image'].data),
                **losses,
                **head_outs
            )
        return outputs
    
    def forward_val(self, **data_batch):
        """
        @description  : 评估的前向推断(原则上是不经过auxiliary的）
        @param  :
        @Returns  :
        """
        losses = dict()
        head_outs = dict()
        features = self.backbone(data_batch["image"] )
        features = self.neck(features)
        for head_key, head in self.decode_heads.items():
            if "trunk" not in head_key:
                continue
            results = head(features, data_batch)
            for key, value in results.items():
                if "out" in key:
                    head_outs[head_key + "_" + key] = value
                if "loss" in key:
                    losses[head_key + "_" + key] = value
                    if "loss" not in losses:
                        losses.setdefault("loss", value)
                    else:
                        losses["loss"] += value
                        
        log_vars = self._parse_losses(losses)
        outputs = dict(
                log_vars=log_vars,
                num_samples=len(data_batch['image'].data),
                **losses,
                **head_outs
            )
        return outputs
    
    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        # 对各种loss进行平均
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return log_vars
        
        
    
    
