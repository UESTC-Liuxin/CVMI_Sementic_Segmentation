'''
Author: Liu Xin
Date: 2021-11-21 21:50:20
LastEditors: Liu Xin
LastEditTime: 2021-11-25 12:19:32
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/epoch_base_runner.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from numpy.lib.shape_base import dsplit
import torch
from .base_runner import BaseRunner
from .builder import RUNNERS
from utils.runner.hooks import HOOKS, Hook
from utils.registry import Registry, build
from utils.static_common_utils import get_host_info, symlink
from utils.runner.checkpoint import save_checkpoint

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self, device,  *args, **kwargs):
        super(EpochBasedRunner, self).__init__(*args, **kwargs)
        self.device = device
        
    def to_device(self,tensors):
        """
        @description  : 将字典中的tensor转移到device上
        @param  :
        @Returns  :
        """
        cuda_tensors={}
        for key,value in tensors.items():
            if(isinstance(value,torch.Tensor)):
                value=value.to(self.device)
            cuda_tensors[key]=value
        return cuda_tensors
    
    def run_iter(self, data_batch, train_mode, **kwargs):
        data_batch = self.to_device(data_batch)
        if train_mode:
            outs = self.model(data_batch, self.optimizer, train_mode, 
                                            **kwargs)
        else:
            outs = self.model(data_batch, self.optimizer, train_mode, **kwargs)
        outputs = dict(data_batch, **outs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        save_history=False,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
        # 如果不保存历史，就只存最新结果
        if not save_history:
            filename = filename_tmpl.format("latest")
        else:
            filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # 保存最优模型
        if self.log_buffer.output.get("is_best", False):
            shutil.copy(filepath, filepath + ".best")
        # 将模型软连接到
        if create_symlink:
            symlink(filename, osp.join(out_dir, 'latest.pth'))
        
    
    def register_criterions_hook(self, criterions_config):
        """
        @description  :注册计算损失的钩子函数
        @param  :
        @Returns  :
        """
        assert criterions_config is not None
        if isinstance(criterions_config, dict):
            criterions_config.setdefault('name', 'LossCaculatorHook')
            hook = build(criterions_config, HOOKS)
        else:
            hook = criterions_config
        self.register_hook(hook, priority='HIGHEST')
    
    def register_evaluate_hook(self, evaluate_config):
        """
        @description  :注册评估的钩子函数
        @param  :
        @Returns  :
        """
        assert evaluate_config is not None
        if isinstance(evaluate_config, dict):
            evaluate_config.setdefault('name', 'EvaluatorHook')
            hook = build(evaluate_config, HOOKS)
        else:
            hook = evaluate_config
        self.register_hook(hook, priority='HIGH')

@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
