'''
Author: Liu Xin
Date: 2021-11-18 16:26:49
LastEditors: Liu Xin
LastEditTime: 2021-11-18 16:27:51
Description: file content
FilePath: /CVMI_Sementic_Segmentation/tools/trainer.py
'''
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer(object):
    
    def __init__(self,args,dataloader:DataLoader,model:nn.Module,optimizer,
                 criterion,logger,summary=None):
        self.args=args
        self.dataloader=dataloader
        self.model = model
        self.logger=logger
        self.summary=summary
        self.criterion=criterion
        self.optimizer=optimizer
        self.start_epoch=0
        # Define lr scheduler
        self.scheduler = LR_Scheduler('poly', args.lr,args.max_epochs, len(self.dataloader))
        #进行训练恢复
        if(args.resume):
            self.resume()

    def resume(self):
        self.logger.info("---------------resume beginning....")
        checkpoint=torch.load(self.args.resume)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        self.start_epoch=checkpoint['epoch']
        self.logger.info("---------------resume end....")

    def dict_to_cuda(self,tensors):
        cuda_tensors={}
        for key,value in tensors.items():
            if(isinstance(value,torch.Tensor)):
                value=value.cuda()
            cuda_tensors[key]=value
        return cuda_tensors

    def train_one_epoch(self,epoch,writer,best_pred):
        self.model.train()
        total_batches = len(self.dataloader)
        tloss = []
        pbar=tqdm(self.dataloader,ncols=100)
        for iter, batch in enumerate(pbar):
            pbar.set_description("Training Processing epoach:{}".format(epoch))
            self.scheduler(self.optimizer, iter, epoch, best_pred)
            batch=self.dict_to_cuda(batch)
            output=self.model(batch)
            loss = self.criterion(output,batch['label'])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tloss.append(loss.item())
            if (iter % self.args.show_interval == 0):
                pred=np.asarray(np.argmax(output['trunk_out'][0].cpu().detach(), axis=0), dtype=np.uint8)
                gt = batch['label'][0]  #每次显示第一张图片
                img = batch['image'][0]  # 每次显示第一张图片
                gt=np.asarray(gt.cpu(), dtype=np.uint8)
                img= np.asarray(img.cpu(), dtype=np.uint8)
                self.visualize(gt,img, pred, epoch*1000+iter,writer,"train")
        