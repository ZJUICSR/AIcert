import os
import pickle
import h5py
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from models import basenet
from models import dataloader
from models.celeba_core import CelebaModel
import utils

class CelebaDomainDiscriminative(CelebaModel):
    def __init__(self, opt):
        super(CelebaDomainDiscriminative, self).__init__(opt)
        
    def _criterion(self, output, target):
        domain_label = target[:, -1:]
        two_n_target = torch.cat([target[:, :-1]*domain_label, 
                                  target[:, :-1]*(1-domain_label)],
                                 dim=1)
        return F.binary_cross_entropy_with_logits(output, two_n_target)
    
    def inference(self, output):
        """Inference method: sum the probability from two domains"""
        
        predict_prob = torch.sigmoid(output).cpu().numpy()
        class_num = predict_prob.shape[1] // 2
        return predict_prob[:, :class_num] + predict_prob[:, class_num:]
        