'''
Author: Liu Xin
Date: 2021-11-21 20:43:20
LastEditors: Liu Xin
LastEditTime: 2021-11-21 20:50:13
Description: file content
FilePath: /CVMI_Sementic_Segmentation/utils/runner/builder.py
'''
import copy

from utils.registry import Registry, build

RUNNERS = Registry('runner')
RUNNER_BUILDERS = Registry('runner builder')


def build_runner_constructor(cfg):
    return build(cfg, RUNNER_BUILDERS)


def build_runner(cfg, default_args=None):
    runner_cfg = copy.deepcopy(cfg)
    constructor_type = runner_cfg.pop('constructor',
                                      'DefaultRunnerConstructor')
    runner_constructor = build_runner_constructor(
        dict(
            type=constructor_type,
            runner_cfg=runner_cfg,
            default_args=default_args))
    runner = runner_constructor()
    return runner
