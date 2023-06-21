import os
import pickle
import h5py
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

class CelebaWeighting(CelebaModel):
    def __init__(self, opt):
        super(CelebaWeighting, self).__init__(opt)
    
    def set_data(self, opt):
        """Set up the dataloaders"""
        data_setting = opt['data_setting']

        # normalize according to ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        image_feature = h5py.File(data_setting['image_feature_path'], 'r')
        target_dict = utils.load_pkl(data_setting['target_dict_path'])
        train_key_list = utils.load_pkl(data_setting['train_key_list_path'])
        dev_key_list = utils.load_pkl(data_setting['dev_key_list_path'])
        test_key_list = utils.load_pkl(data_setting['test_key_list_path'])
        
        train_data = dataloader.CelebaDatasetWithWeight(train_key_list, image_feature, 
                                                        target_dict, transform_train)
        self.train_loader = torch.utils.data.DataLoader(
                                train_data, batch_size=opt['batch_size'],
                                shuffle=True, num_workers=1)
        dev_data = dataloader.CelebaDatasetWithWeight(dev_key_list, image_feature, 
                                                       target_dict, transform_test)
        self.dev_loader = torch.utils.data.DataLoader(
                              dev_data, batch_size=opt['batch_size'],
                              shuffle=False, num_workers=1)
        test_data = dataloader.CelebaDatasetWithWeight(test_key_list, image_feature, 
                                                       target_dict, transform_test)
        self.test_loader = torch.utils.data.DataLoader(
                               test_data, batch_size=opt['batch_size'],
                               shuffle=False, num_workers=1)
        
        self.dev_target = np.array([target_dict[key] for key in dev_key_list])
        self.dev_class_weight = utils.compute_class_weight(self.dev_target)
        self.test_target = np.array([target_dict[key] for key in test_key_list])
        self.test_class_weight = utils.compute_class_weight(self.test_target)
        self.subclass_idx = utils.load_pkl(data_setting['subclass_idx_path'])
        
    def _criterion(self, output, target, weight):
        loss = 0.
        for i in range(target.size(1)-1):
            loss += F.binary_cross_entropy_with_logits(
                        output[:, i], target[:, i], weight[:, i])
        return loss
    
    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        
        train_loss = 0
        for i, (images, targets, weights) in enumerate(loader):
            images, targets, weights = \
                images.to(self.device), targets.to(self.device), weights.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss = self._criterion(outputs, targets, weights)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.log_result('Train iteration', {'loss': loss.item()},
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}'.format(
                      self.epoch, i+1, len(loader), loss.item()))
        
        self.log_result('Train epoch', {'loss': train_loss/len(loader)}, self.epoch)
        self.epoch += 1
        
    def _test(self, loader):
        """Compute model output on test set"""
        
        self.network.eval()

        test_loss = 0
        output_list = []
        feature_list = []
        with torch.no_grad():
            for i, (images, targets, weights) in enumerate(loader):
                images, targets, weights = \
                    images.to(self.device), targets.to(self.device), weights.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets, weights)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)

        return test_loss, torch.cat(output_list), torch.cat(feature_list)
    
    