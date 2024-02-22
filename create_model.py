import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import json
import numpy as np
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from collections import OrderedDict
from torchvision.transforms import v2
torch.set_float32_matmul_precision('high')


class conv_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes=1):
        super(conv_head, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 128, (3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding=(1,1)),
        )
    def forward(self, x):
        x = self.net(x)
        return x


class SegmentationNet(nn.Module):
    def __init__(self, backbone, head, embedding_size, patch_size):
        super(SegmentationNet, self).__init__()

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.head = head(self.embedding_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        with torch.no_grad():
            x = self.backbone.forward_features(x)
            x = x['x_norm_patchtokens']
            x = x.permute(0,2,1)
            x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))
        x = self.head(x)
        return x
    


class LitSegmentator(pl.LightningModule):

    def __init__(self, net, lr, weight_decay):
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = nn.BCEWithLogitsLoss()
        self.train_loss = MeanMetric()
        self.train_precision = torchmetrics.Precision(num_classes=1, task='binary')
        self.train_recall = torchmetrics.Recall(num_classes=1, task='binary')
        self.val_loss = MeanMetric()
        self.val_precision = torchmetrics.Precision(num_classes=1, task='binary')
        self.val_recall = torchmetrics.Recall(num_classes=1, task='binary')
        # self.jaccard_train = BinaryJaccardIndex()
        # self.jaccard_val = BinaryJaccardIndex()
            
    def training_step(self, batch, batch_idx):

        x, target = batch
        predictions = self.net(x)
        predictions = torch.squeeze(predictions)
        loss = self.loss(predictions, target)
        self.train_loss(loss)
        t = torch.Tensor([0.5]).cuda()  # threshold
        predictions_bin = (predictions > t).to(int)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output
    

    def validation_step(self, batch, batch_idx):
        x, target = batch
        predictions = self.net(x)
        predictions = torch.squeeze(predictions)
        loss = self.loss(predictions, target)
        
        self.val_loss(loss)
        t = torch.Tensor([0.5]).cuda()  # threshold
        predictions_bin = (predictions > t).to(int)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches)
        
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'name': 'train/lr',
                    'scheduler': scheduler,
                    'interval': 'step', 
                    'frequency': 1,
                    }
                }
    
    def forward(self, x):
        x = self.net(x)
        return x
    
