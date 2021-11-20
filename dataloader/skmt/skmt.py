# -*- encoding: utf-8 -*-
'''
@File    :   skmt.py
@Author  :   Liu Xin 
@CreateTime    :   2021/11/15 19:18:28
@Version :   1.0
@Contact :   xinliu1996@163.com
@License :   (C)Copyright 2020-2025 CVMI(UESTC)
@Desc    :   None
'''


from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import collections
from torch.utils.data import Dataset
from torchvision import transforms
from dataloader.transforms_utils import custom_transforms as tr
from dataloader.transforms_utils import augment as au
from dataloader.transforms_utils import meta_transforms as meta_t
from dataloader.builder import DATASET

@DATASET.register_module("skmt")
class SkmtDataSet(Dataset):
    """
    PascalVoc dataset
    """
    CLASSES = ('background', 'SAS', 'LHB', 'D',
               'HH', 'SUB', 'SUP', 'GL', 'GC',
               'SCB', 'INF', 'C', 'TM', 'SHB',
               'LHT', 'SAC', 'INS', 'BBLH', 'LHBT')

    PALETTE = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                          [128, 0, 128], [0, 128, 128], [
                              128, 128, 128], [64, 0, 0],
                          [192, 0, 0], [64, 128, 0]])

    CLASSES_PIXS_WEIGHTS = (0.7450, 0.0501, 0.0016, 0.0932, 0.0611,
                            0.0085, 0.0092, 0.0014, 0.0073, 0.0012, 0.0213)

    # TODO:取消未出现的类
    # NUM_CLASSES = len(CLASSES)
    NUM_CLASSES = 11

    def __init__(self, dataset_root, transforms, split='train', **kwargs):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._dataset_root = dataset_root
        self.transforms = transforms
        self.split = split
        self._image_dir = os.path.join(self._dataset_root, 'JPEGImages')
        self._mask_dir = os.path.join(self._dataset_root, 'SegmentationClass')
        _splits_dir = os.path.join(self._dataset_root, 'ImageSets')

        self.im_ids = []
        self.images = []
        self.masks = []

        # 读入
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()
        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".jpg")
            _mask = os.path.join(self._mask_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_mask)
            self.im_ids.append(line)
            self.images.append(_image)
            self.masks.append(_mask)
                
        assert (len(self.images) == len(self.masks))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        _section = self.get_section(index)
        sample = {'image': _img, 'label': _target, 'section': _section}
        if self.split == "train":
            for key, value in self.transforms(sample).items():
                sample[key] = value
            return sample
        else:
            for key, value in self.transforms(sample).items():
                sample[key] = value
            return sample

    def get_section(self, index):
        _name = self.images[index].split('/')[-1]
        _section = _name.split('_')[0][-2]
        return int(_section)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])
        return _img, _target

    def count_section(self):
        """count the section num
        :param img_list:
        :return:
        """
        table = collections.OrderedDict()  # 将普通字典转换为有序字典
        for i in range(len(self.images)):
            _section = self.get_section(i)
            if(_section not in table.keys()):
                table[_section] = 0
            table[_section] = table[_section]+1
        return table

    @classmethod
    def encode_segmap(cls, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

        for ii, label in enumerate(cls.PALETTE):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    @classmethod
    def decode_segmap(cls, label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = cls.PALETTE
        n_classes = len(label_colours)

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def __str__(self):
        return 'skmt(split=' + str(self.split) + ')'


