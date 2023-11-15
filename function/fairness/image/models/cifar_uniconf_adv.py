import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from function.fairness.image.models import basenet
from function.fairness.image.models import dataloader
from function.fairness.image.models.cifar_core import CifarModel
from function.fairness.image import utils

class CifarUniConfAdv(CifarModel):
    def __init__(self, opt):
        super(CifarUniConfAdv, self).__init__(opt)
        self.training_ratio = opt['training_ratio']
        self.alpha = opt['alpha']
        
    def set_network(self, opt):
        """Define the network"""
        
        self.base_network = basenet.ResNet18_base().to(self.device)
        
        # Two fc layers on top of the base network, one for target classification,
        # one for domain classification
        self.class_network = nn.Linear(512, opt['output_dim']).to(self.device)
        self.domain_network = nn.Linear(512, 2).to(self.device)
        
    def set_data(self, opt):
        """Set up the dataloaders"""
        
        data_setting = opt['data_setting']

        with open(data_setting['train_data_path'], 'rb') as f:
            train_array = pickle.load(f)

        mean = tuple(np.mean(train_array / 255., axis=(0, 1, 2)))
        std = tuple(np.std(train_array / 255., axis=(0, 1, 2)))
        normalize = transforms.Normalize(mean=mean, std=std)

        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_data = dataloader.CifarDatasetWithDomain(data_setting['train_data_path'], 
                                             data_setting['train_label_path'],
                                             data_setting['domain_label_path'],
                                             transform_train)
        test_color_data = dataloader.CifarDatasetWithDomain(data_setting['test_color_path'], 
                                                  data_setting['test_label_path'],
                                                  data_setting['domain_label_path'],
                                                  transform_test)
        test_gray_data = dataloader.CifarDatasetWithDomain(data_setting['test_gray_path'], 
                                                 data_setting['test_label_path'],
                                                 data_setting['domain_label_path'],
                                                 transform_test)

        self.test_color_target = np.array(test_color_data.class_label)
        self.test_gray_target = np.array(test_gray_data.class_label)

        self.train_loader = torch.utils.data.DataLoader(
                                 train_data, batch_size=opt['batch_size'],
                                 shuffle=True, num_workers=1)
        self.test_color_loader = torch.utils.data.DataLoader(
                                      test_color_data, batch_size=opt['batch_size'],
                                      shuffle=False, num_workers=1)
        self.test_gray_loader = torch.utils.data.DataLoader(
                                     test_gray_data, batch_size=opt['batch_size'],
                                     shuffle=False, num_workers=1)
        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.base_optimizer = optimizer_setting['optimizer']( 
                                  params=self.base_network.parameters(), 
                                  lr=optimizer_setting['lr'],
                                  weight_decay=optimizer_setting['weight_decay']
                                 )
        self.class_optimizer = optimizer_setting['optimizer']( 
                                  params=self.class_network.parameters(), 
                                  lr=optimizer_setting['lr'],
                                  weight_decay=optimizer_setting['weight_decay']
                                 )
        self.domain_optimizer = optimizer_setting['optimizer']( 
                                  params=self.domain_network.parameters(), 
                                  lr=optimizer_setting['lr'],
                                  weight_decay=optimizer_setting['weight_decay']
                                 )
        
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
    
    def  _train(self, loader):
        """Train the model for one epoch"""
        
        self.base_network.train()
        self.class_network.train()
        self.domain_network.train()
        
        train_class_loss = 0
        train_domain_loss = 0
        total = 0
        class_correct = 0
        domain_correct = 0
        
        for i, (images, class_labels, domain_labels) in enumerate(loader):
            images, class_labels, domain_labels = images.to(self.device), \
                class_labels.to(self.device), domain_labels.to(self.device)
            
            self.base_optimizer.zero_grad()
            self.class_optimizer.zero_grad()
            self.domain_optimizer.zero_grad()
            
            features = self.base_network(images)
            class_outputs = self.class_network(features)
            domain_outputs = self.domain_network(features)
            
            class_loss = self._criterion(class_outputs, class_labels)
            domain_loss = self._criterion(domain_outputs, domain_labels)
            
            if self.epoch % self.training_ratio == 0:
                # Update the main network
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
                
            total += class_labels.size(0)
            train_class_loss +=  class_loss.item()
            _, class_predicted = class_outputs.max(1)
            class_correct += class_predicted.eq(class_labels).sum().item()
            
            train_domain_loss += domain_loss.item()
            _, domain_predicted = domain_outputs.max(1)
            domain_correct += domain_predicted.eq(domain_labels).sum().item()
            
            train_result = {
                'class_loss': class_loss.item(),
                'domain_loss': domain_loss.item(),
                'class_accuracy': 100.*class_correct/total,
                'domain_accuracy': 100.*domain_correct/total
            }
            
            self.log_result('Train_iteraion', train_result,
                            len(loader)*self.epoch + i)
            
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch: {} [{}|{}], class loss:{}, class accuracy:{}, '\
                      'domain loss: {}, domain accuracy: {}'.format(
                      self.epoch, i+1, len(loader), class_loss, 100.*class_correct/total,
                      domain_loss, 100.*domain_correct/total))
            
        self.epoch += 1
        
    def _test(self, loader):
        """Test the model performance"""
        
        self.base_network.eval()
        self.class_network.eval()
        self.domain_network.eval()
        
        total = 0 
        class_correct = 0
        domain_correct = 0
        test_class_loss = 0
        test_domain_loss = 0
        feature_list = []
        class_output_list = []
        domain_output_list = []
        class_predict_list = []
        domain_predict_list = []
        
        with torch.no_grad():
            for i, (images, class_labels, domain_labels) in enumerate(loader):
                images, class_labels, domain_labels = images.to(self.device), \
                    class_labels.to(self.device), domain_labels.to(self.device)
                features = self.base_network(images)
                class_outputs = self.class_network(features)
                domain_outputs = self.domain_network(features)
                
                class_loss = self._criterion(class_outputs, class_labels)
                domain_loss = self._criterion(domain_outputs, domain_labels)
                
                test_class_loss += class_loss.item()
                test_domain_loss += domain_loss.item()
                
                total += class_labels.size(0)
                _, class_predicted = class_outputs.max(1)
                class_correct += class_predicted.eq(class_labels).sum().item()
                
                _, domain_predicted = domain_outputs.max(1)
                domain_correct += domain_predicted.eq(domain_labels).sum().item()
                
                class_predict_list.extend(class_predicted.tolist())
                class_output_list.append(class_outputs.cpu().numpy())
                domain_output_list.append(domain_outputs.cpu().numpy())
                feature_list.append(features.cpu().numpy())
                
        test_result = {
            'class_loss': test_class_loss/len(loader),
            'domain_loss': test_domain_loss/len(loader),
            'class_accuracy': 100.*class_correct/total,
            'domain_accuracy': 100.*domain_correct/total,
            'class_predict_labels': class_predict_list,
            'domain_predict_labels': domain_predict_list,
            'class_outputs': np.vstack(class_output_list),
            'domain_outputs': np.vstack(domain_output_list),
            'features': np.vstack(feature_list),
            'predict_labels': np.array(class_predict_list),
        }
        
        return test_result
                
    def train(self):
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        
    def test(self):
        # Test and save the result
        state_dict = None
        if os.path.exists(os.path.join(self.save_path, 'best.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'best.pth'))
        elif os.path.exists(os.path.join(self.save_path, 'ckpt.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        # elif os.path.exists(self.model_path):
        #     state_dict = torch.load(self.model_path)
        else:
            raise FileNotFoundError("no checkpoints available for testing")
        self.load_state_dict(state_dict)
        
        test_color_result = self._test(self.test_color_loader)
        test_gray_result = self._test(self.test_gray_loader)
        utils.save_pkl(test_color_result, os.path.join(self.save_path, 'test_color_result.pkl'))
        utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))
        
        # Output the classification accuracy on test set
        info = ('Test on color images accuracy: {}, domain accuracy; {}\n' 
                'Test on gray images accuracy: {}, domain accuracy: {}'
                .format(test_color_result['class_accuracy'], test_color_result['domain_accuracy'],
                        test_gray_result['class_accuracy'], test_gray_result['domain_accuracy']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
        result = self.trans_result(test_color_result, test_gray_result)
        return result