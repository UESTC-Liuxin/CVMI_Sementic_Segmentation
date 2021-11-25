'''
Author: Liu Xin
Date: 2021-11-22 21:45:56
LastEditors: Liu Xin
LastEditTime: 2021-11-25 16:05:18
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/hooks/logger/tensorboard.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from subprocess import run
from utils.ddp.dist_utils import master_only
from utils.runner.hooks import HOOKS
from utils.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorboardX to use '
                                'TensorboardLoggerHook.')
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_epoch(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_epoch(runner))
        info_dict = dict(
            train_mode=runner.mode,
            epoch=self.get_epoch(runner),
            result=tags
        )
        formated_info = [f"{key}:{value}"  for key, value in info_dict.items()]
        runner.logger.info('Running:\n%s', "\n".join(formated_info))

    @master_only
    def after_run(self, runner):
        self.writer.close()
