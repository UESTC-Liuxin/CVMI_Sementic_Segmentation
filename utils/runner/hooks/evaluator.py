'''
Author: Liu Xin
Date: 2021-11-23 10:18:26
LastEditors: Liu Xin
LastEditTime: 2021-11-29 21:59:39
Description: 评估指标evaluation hook
FilePath: /CVMI_Sementic_Segmentation/utils/runner/hooks/evaluator.py
'''
from PIL import Image
import numpy as np
from utils.runner.hooks.hook import HOOKS, Hook
from utils.ddp.dist_utils import master_only, collect_results_gpu
from utils.ddp.dist_utils import  get_dist_info
@HOOKS.register_module()
class EvaluatorHook(Hook):
    """计算loss的钩子函数

    Args:
        Hook ([type]): [description]
    """
    def __init__(self, evaluator, *args, **kwargs) :
        super(Hook, self).__init__(*args, **kwargs)
        self.evaluator = evaluator
        
    def after_val_iter(self, runner):
        """
        @description  : 每个iter之后需要对结果进行同步
        @param  :
        @Returns  :
        """
        rank , _ = get_dist_info()
        mask = runner.outputs["mask"]
        pred = runner.outputs["trunk_seg_head_seg_out"]
        mask = np.asarray(mask.cpu().detach().squeeze(0), dtype=np.uint8)
        pred = np.asarray(np.argmax(pred.cpu().detach(), axis=1), dtype=np.uint8)
        matrix = self.evaluator.get_matrix(mask, pred)
        collect_result = collect_results_gpu([matrix])
        if collect_result is None:
            return
        self.evaluator.add_batch(*collect_result)
        self.write_mask(runner, mask, pred, rank)
        
    def write_mask(self, runner, mask,pred, rank = 0):
        batch_size = len(mask)
        inner_iter = runner.inner_iter
        for i in range(batch_size):
            mask_pred = np.concatenate([mask[i], pred[i]], axis=-1)
            rgb = runner.data_loader.dataset.decode_segmap(mask_pred)
            rgb = (255 * rgb).astype(np.uint8)
            image = Image.fromarray(rgb)
            image.save(f"/home/liuxin/Documents/CVMI_Sementic_Segmentation/result/{inner_iter}_{rank}_{i}.png")
        
    def before_val_epoch(self, runner):
        """
        @description  : 训练前需要重置evaluator
        @param  :
        @Returns  :
        """
        self.evaluator.reset()
        
    @master_only  
    def after_val_epoch(self, runner):
        """
        @description  :  获取整个结果
        @param  :
        @Returns  :
        """
        evaluate_result = self.evaluator.get_specify_result()
        for name, val in evaluate_result.items():
                runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        