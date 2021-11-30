'''
Author: Liu Xin
Date: 2021-11-29 11:08:53
LastEditors: Liu Xin
LastEditTime: 2021-11-30 19:43:19
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/decode_heads/encnet/encnet.py
'''
'''
Author: Liu Xin
Date: 2021-11-29 11:08:53
LastEditors: Liu Xin
LastEditTime: 2021-11-30 19:31:07
Description: file content
FilePath: /CVMI_Sementic_Segmentation/model/decode_heads/encnet/encnet.py
'''

"""Context Encoding for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.enc_module import EncModule
from model.builder import DECODE_HEAD

__all__ = ['EncNet']


@DECODE_HEAD.register_module("EncNet")
class EncNet(nn.Module):
    def __init__(self, in_channels, num_classes, criterion, match_block,lateral=True,**kwargs):
        super(EncNet, self).__init__()
        self.head = _EncHead(in_channels, num_classes, lateral=lateral, **kwargs)
        self.match_block = match_block
        self.criterion = criterion
        self.__setattr__('exclusive', ['head'])

    def forward(self, inputs, data_batch):
        base_out, se_out = self.head(*inputs) 
        out = self.match_block(base_out)
        seg_loss, se_loss = self.criterion(out, se_out, data_batch["mask"])
        return {"seg_out":out, " seg_loss":seg_loss, "se_loss":se_loss}


class _EncHead(nn.Module):
    def __init__(self, in_channels, num_classes, se_loss=True, lateral=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, 1, bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, 1, bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)),
            ])
            self.fusion = nn.Sequential(
                nn.Conv2d(3 * 512, 512, 3, padding=1, bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            )
        self.encmodule = EncModule(512, num_classes, ncodes=32, se_loss=se_loss,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1, False),
            nn.Conv2d(512, num_classes, 1)
        )

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


if __name__ == '__main__':
    x1 = torch.randn(4,256,64,64)
    x2 = torch.randn(4,512,16,16)
    x3 = torch.randn(4,1024,16,16)
    x4 = torch.randn(4,2048,16,16)
    
    model = EncNet(2048,11)
    out = model([x1,x2,x3,x4])
    print(type(out))
    # outputs = model(img)
