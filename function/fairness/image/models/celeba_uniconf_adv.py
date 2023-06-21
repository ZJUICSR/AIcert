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

class CelebaUniConfAdv(CelebaModel):
    def __init__(self, opt):
        super(CelebaUniConfAdv, self).__init__(opt)
        self.training_ratio = opt['training_ratio']
        self.alpha = opt['alpha']
        
    def set_network(self, opt):
        """Define the network"""
        
        self.base_network = basenet.ResNet50_base(pretrained=True,
                                                  dropout=opt['dropout']).to(self.device)
        
        # Two fc layers on top of the base network, one for target classification,
        # one for domain classification
        self.class_network = nn.Linear(2048, opt['output_dim']).to(self.device)
        self.domain_network = nn.Linear(2048, 2).to(self.device)
        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.base_optimizer = optimizer_setting['optimizer']( 
                                  params=filter(lambda p: p.requires_grad, self.base_network.parameters()), 
                                  lr=optimizer_setting['lr'],
                                  weight_decay=optimizer_setting['weight_decay'])
        self.class_optimizer = optimizer_setting['optimizer']( 
                                  params=filter(lambda p: p.requires_grad, self.class_network.parameters()), 
                                  lr=optimizer_setting['lr'],
                                  weight_decay=optimizer_setting['weight_decay'])
        self.domain_optimizer = optimizer_setting['optimizer']( 
                                  params=filter(lambda p: p.requires_grad, self.domain_network.parameters()), 
                                  lr=optimizer_setting['lr'],
                                  weight_decay=optimizer_setting['weight_decay'])
        
    def _domain_criterion(self, output, target):
        return F.cross_entropy(output, target[:, -1].long())
    
    def state_dict(self):
        state_dict = {
            'base_network': self.base_network.state_dict(),
            'class_network': self.class_network.state_dict(),
            'domain_network': self.domain_network.state_dict(),
            'base_optimizer': self.base_optimizer.state_dict(),
            'class_optimizer': self.class_optimizer.state_dict(),
            'domain_optimizer': self.domain_optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict
        
    def load_state_dict(self, state_dict):
        self.base_network.load_state_dict(state_dict['base_network'])
        self.class_network.load_state_dict(state_dict['class_network'])
        self.domain_network.load_state_dict(state_dict['domain_network'])
    
    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.base_network.train()
        self.class_network.train()
        self.domain_network.train()
        
        train_class_loss = 0
        train_domain_loss = 0
        total = 0
        correct = 0
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.base_optimizer.zero_grad()
            self.class_optimizer.zero_grad()
            self.domain_optimizer.zero_grad()
            
            features = self.base_network(images)
            class_outputs = self.class_network(features)
            domain_outputs = self.domain_network(features)
            
            class_loss = self._criterion(class_outputs, targets)
            domain_loss = self._domain_criterion(domain_outputs, targets)
            
            total += targets.size(0)
            _,predicted = domain_outputs.max(1)
            correct += predicted.eq(targets[:, -1].long()).sum().item()
            
            # Update the main network
            if self.epoch % self.training_ratio == 0:
                log_softmax = F.log_softmax(domain_outputs, dim=1)
                confusion_loss = -log_softmax.mean(dim=1).mean()
                loss = class_loss + self.alpha*confusion_loss
                loss.backward()
                self.class_optimizer.step()
                self.base_optimizer.step()
            else:
                # Update the domain classifier
                domain_loss.backward()
                self.domain_optimizer.step()

            train_class_loss += class_loss.item()
            train_domain_loss += domain_loss.item()
            self.log_result('Train iteration', 
                            {'class_loss': class_loss.item(),
                             'domain_loss': domain_loss.item(),
                             'domain_accuracy': 100.*correct/total},
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], class loss:{}, domain loss: {}, domain accuracy: {}'
                      .format(self.epoch, i+1, len(loader), 
                              class_loss.item(), domain_loss.item(),
                              100.*correct/total))
        
        self.log_result('Train epoch', 
                        {'class_loss': train_class_loss/len(loader),
                         'domain_loss': train_domain_loss/len(loader),
                         'domain_accuracy': 100.*correct/total}, 
                        self.epoch)
        self.epoch += 1
        
    def _test(self, loader):
        """Compute model output on test set"""
        
        self.base_network.eval()
        self.class_network.eval()
        self.domain_network.eval()
        
        test_class_loss = 0
        test_domain_loss = 0
        total = 0
        correct = 0
        feature_list = []
        class_output_list = []
        domain_output_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                features = self.base_network(images)
                class_outputs = self.class_network(features)
                domain_outputs = self.domain_network(features)
            
                class_loss = self._criterion(class_outputs, targets)
                domain_loss = self._domain_criterion(domain_outputs, targets)
                test_class_loss += class_loss.item()
                test_domain_loss += domain_loss.item()
                
                total += targets.size(0)
                _, predicted = domain_outputs.max(1)
                correct += predicted.eq(targets[:, -1].long()).sum().item() 

                class_output_list.append(class_outputs)
                domain_output_list.append(domain_outputs)
                feature_list.append(features)
                
            return test_class_loss, test_domain_loss, torch.cat(class_output_list), \
                   torch.cat(domain_output_list), torch.cat(feature_list), 100.*correct/total
        
    def train(self):
        """Train the model for one epoch, evaluate on validation set and 
        save the best model
        """
        
        start_time = datetime.now()
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        
        dev_class_loss, dev_domain_loss, dev_class_output, _,_, dev_domain_accuarcy = \
            self._test(self.dev_loader)
        dev_predict_prob = self.inference(dev_class_output)
        dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
                                                     self.dev_class_weight)
        dev_mAP = utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
        
        self.log_result('Dev epoch', 
                        {'class_loss': dev_class_loss/len(self.dev_loader), 
                         'domain_loss': dev_domain_loss/len(self.dev_loader),
                         'mAP': dev_mAP,
                         'domain_accuracy': dev_domain_accuarcy},
                        self.epoch)
        if (self.epoch) > 1 and (dev_mAP > self.best_dev_mAP):
            self.best_dev_mAP = dev_mAP
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best.pth'))
        
        duration = datetime.now() - start_time
        print('Finish training epoch {}, dev class loss: {}, dev doamin loss: {}, dev mAP: {},'\
              'domain_accuracy: {}, time used: {}'
              .format(self.epoch, dev_class_loss/len(self.dev_loader), 
                      dev_domain_loss/len(self.dev_loader), dev_mAP, dev_domain_accuarcy,
                      duration))
        
    def test(self):
        # Test and save the result
        state_dict = None
        if os.path.exists(os.path.join(self.save_path, 'best.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'best.pth'))
        elif os.path.exists(os.path.join(self.save_path, 'ckpt.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        else:
            raise FileNotFoundError("no checkpoints available for testing")

        self.load_state_dict(state_dict)
        
        dev_class_loss, dev_domain_loss, dev_class_output, dev_domain_output, \
            dev_feature, dev_domain_accuracy = self._test(self.dev_loader)
        dev_predict_prob = self.inference(dev_class_output)
        dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
                                                     self.dev_class_weight)
        dev_mAP = utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
        dev_result = {'output': dev_class_output.cpu().numpy(), 
                      'feature': dev_feature.cpu().numpy(),
                      'per_class_AP': dev_per_class_AP,
                      'mAP': dev_mAP,
                      'domain_output': dev_domain_output.cpu().numpy(),
                      'domain_accuracy': dev_domain_accuracy}
        utils.save_pkl(dev_result, os.path.join(self.save_path, 'dev_result.pkl'))
        
        test_class_loss, test_domain_loss, test_class_output, test_domain_output, \
            test_feature, test_domain_accuracy = self._test(self.test_loader)
        test_predict_prob = self.inference(test_class_output)
        test_per_class_AP = utils.compute_weighted_AP(self.test_target, test_predict_prob, 
                                                     self.test_class_weight)
        test_mAP = utils.compute_mAP(test_per_class_AP, self.subclass_idx)
        test_result = {'output': test_class_output.cpu().numpy(), 
                      'feature': test_feature.cpu().numpy(),
                      'per_class_AP': test_per_class_AP,
                      'mAP': test_mAP,
                      'domain_output': test_domain_output.cpu().numpy(),
                      'domain_accuracy': test_domain_accuracy}
        utils.save_pkl(test_result, os.path.join(self.save_path, 'test_result.pkl'))
        
        # Output the mean AP for the best model on dev and test set
        info = ('Dev mAP: {}\n'
                'Test mAP: {}'.format(dev_mAP, test_mAP))
        utils.write_info(os.path.join(self.save_path, 'result.txt'), info)
        result = {
            "y_pred": test_predict_prob[:, self.subclass_idx],
            "y_true": self.test_target[:, self.subclass_idx],
            "z": self.test_target[:, -1]
        }
        return result