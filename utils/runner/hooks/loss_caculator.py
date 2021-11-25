'''
Author: Liu Xin
Date: 2021-11-22 16:42:48
LastEditors: Liu Xin
LastEditTime: 2021-11-23 10:19:42
Description: 计算loss的钩子函数，在mmsegmentation中，loss的计算是放在model中, 但个人觉得不合理
FilePath: /CVMI_Sementic_Segmentation/utils/runner/hooks/loss_caculator.py
'''
from os import name
from subprocess import run
from utils.runner.hooks.hook import HOOKS, Hook

@HOOKS.register_module()
class LossCaculatorHook(Hook):
    """计算loss的钩子函数

    Args:
        Hook ([type]): [description]
    """
    def __init__(self, criterions, *args, **kwargs) :
        super(Hook, self).__init__(*args, **kwargs)
        self.criterions = criterions
        
    def after_iter(self, runner):
        """
        @description  : 对前向后进行损失计算
        @param  :
        @Returns  :
        """
        outputs = runner.outputs
        losses_dict = dict()
        for key, value in outputs.items():
            name = key.replace("_out","")
            if "seg" in name:
                mask = outputs["mask"]
                loss = self.caculate_loss(name, value, mask)
                losses_dict[name + "_loss"] = loss
                if "loss" not in losses_dict:
                    losses_dict.setdefault("loss", loss)
                else:
                    losses_dict["loss"] += loss
                # print(outputs[name + "_loss"])
        outputs.update(losses_dict) 
        
    def after_train_iter(self, runner):
        return self.after_iter(runner)
    
    def after_val_iter(self, runner):
        return self.after_iter(runner)
        

        # print(outputs)
    def caculate_loss(self, name, output, gt):
        """
        @description  :
        @param  :
        @Returns  :
        """
        return self.criterions[name + "_criterion"](output, gt)
        
        
        
        
        