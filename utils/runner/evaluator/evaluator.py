'''
Author: Liu Xin
Date: 2021-11-23 10:24:00
LastEditors: Liu Xin
LastEditTime: 2021-11-25 11:52:52
Description: 评估方法
FilePath: /CVMI_Sementic_Segmentation/utils/runner/evaluator/evaluator.py
'''
import os
import sys
from collections import OrderedDict
import numpy as np
from utils.runner.evaluator.builder import EVALUATOR
from utils.ddp.dist_utils import master_only


@EVALUATOR.register_module()
class BaseEvaluator(object):
    def __init__(self, num_classes, evaluate_list, best_key = "mIou"):
        self.num_class = num_classes
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.evaluate_list = evaluate_list
        self.best_epoch = 0
        self.best_key = best_key
        self.best_result = OrderedDict()
        self.is_best = False
        assert best_key in evaluate_list
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
    def get_specify_result(self):
        result = dict()
        for name in self.evaluate_list:
            result[name] = getattr(self, name)
        result.update(is_best = self.is_best)
        return result
    
    def set_best(self, key, value):
        best_temp = self.best_result.get(key, -1)
        assert isinstance(value, float)
        is_best = best_temp < value
        if is_best:
            self.best_result[key] = value
        if self.best_key == key:
            self.is_best = (is_best == True)
        
    def _generate_matrix(self, mask, pred_image):
        bool_mask_ = (mask >= 0) & (mask < self.num_class)
        label = self.num_class * mask[bool_mask_].astype('int') + pred_image[bool_mask_]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
     
    @property
    def overall_acc(self):
        """
        caculate Overall Acc
        :return:
        """
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        self.set_best("overall_acc", acc)
        return acc

    @property
    def macc(self):
        """
        caculate mean acc by class
        :return:
        """
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        self.set_best("macc", acc)
        return acc

    @property
    def mIou(self):
        """
        caculate mean Iou
        :return:
        """
        mIou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        mIou = np.nanmean(mIou)
        self.set_best("mIou", mIou)
        return mIou
    
    @property
    def mDice(self):
        mDice = (2 * np.diag(self.confusion_matrix)) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        )
        mDice = np.nanmean(mDice)
        self.set_best("mIou", mDice)
        return mDice

    @property
    def fwIou(self):
        """
        caculate frequency weighted intersection over union
        :return:
        """        
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        fwIou = (freq[freq > 0] * iu[freq > 0]).sum()
        self.set_best("fwIou", fwIou)
        return fwIou
    
    @property
    def acc_cls(self):
        acc_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc_cls
    
    @property
    def iou_cls(self):
        iou_cls=np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return iou_cls



    @master_only
    def add_batch(self, mask, pred_image):
        assert mask.shape == pred_image.shape
        self.confusion_matrix += self._generate_matrix(mask, pred_image)

if __name__ == "__main__":
    evaluator = BaseEvaluator(11, ["mIou"])
    evaluator.reset()
    mask = np.random.randint(0, 10, size=(32,256,256))
    evaluator.add_batch(mask,mask)
    print(evaluator.mIou())