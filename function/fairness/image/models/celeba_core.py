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
from image.models import basenet
from image.models import dataloader
from image import utils

class CelebaModel():
    def __init__(self, opt):
        super(CelebaModel, self).__init__()
        self.epoch = 0
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))
        
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)
        self.best_dev_mAP = 0.

    def set_network(self, opt):
        """Define the network"""
        
        self.network = basenet.ResNet50(n_classes=opt['output_dim'],
                                        pretrained=True,
                                        dropout=opt['dropout']).to(self.device)
        
    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

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
        
        train_data = dataloader.CelebaDataset(train_key_list, image_feature, 
                                              target_dict, transform_train)
        self.train_loader = torch.utils.data.DataLoader(
                                train_data, batch_size=opt['batch_size'],
                                shuffle=True, num_workers=0)
        dev_data = dataloader.CelebaDataset(dev_key_list, image_feature, 
                                            target_dict, transform_test)
        self.dev_loader = torch.utils.data.DataLoader(
                              dev_data, batch_size=opt['batch_size'],
                              shuffle=False, num_workers=0)
        test_data = dataloader.CelebaDataset(test_key_list, image_feature, 
                                             target_dict, transform_test)
        self.test_loader = torch.utils.data.DataLoader(
                               test_data, batch_size=opt['batch_size'],
                               shuffle=False, num_workers=0)
        
        self.dev_target = np.array([target_dict[key] for key in dev_key_list])
        self.dev_class_weight = utils.compute_class_weight(self.dev_target)
        self.test_target = np.array([target_dict[key] for key in test_key_list])
        self.test_class_weight = utils.compute_class_weight(self.test_target)
        
        # We only evaluate on a subset of attributes that have enough test data
        # on both domain
        self.subclass_idx = utils.load_pkl(data_setting['subclass_idx_path'])
        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer']( 
                            params=filter(lambda p: p.requires_grad, self.network.parameters()), 
                            lr=optimizer_setting['lr'],
                            weight_decay=optimizer_setting['weight_decay']
                            )
        
    def _criterion(self, output, target):
        return F.binary_cross_entropy_with_logits(output, target[:, :-1])
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        
        train_loss = 0
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss = self._criterion(outputs, targets)
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
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)

        return test_loss, torch.cat(output_list), torch.cat(feature_list)

    def inference(self, output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().numpy()
    
    def train(self):
        """Train the model for one epoch, evaluate on validation set and 
        save the best model
        """
        
        # save a ckpt for testing
        # utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        
        start_time = datetime.now()
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        
        dev_loss, dev_output, _ = self._test(self.dev_loader)
        dev_predict_prob = self.inference(dev_output)
        dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
                                                     self.dev_class_weight)
        dev_mAP = utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
        
        self.log_result('Dev epoch', {'loss': dev_loss/len(self.dev_loader), 'mAP': dev_mAP},
                        self.epoch)
        if dev_mAP > self.best_dev_mAP:
            self.best_dev_mAP = dev_mAP
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best.pth'))
        
        duration = datetime.now() - start_time
        print('Finish training epoch {}, dev mAP: {}, time used: {}'.format(self.epoch, dev_mAP, duration))

    def test(self):
        # Test and save the result
        state_dict = None
        if os.path.exists(os.path.join(self.save_path, 'best.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'best.pth'))
        elif os.path.exists(os.path.join(self.save_path, 'ckpt.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        else:
            raise FileNotFoundError("no checkpoints available for testing")

        self.network.load_state_dict(state_dict['model'])
        
        # dev_loss, dev_output, dev_feature = self._test(self.dev_loader)
        # dev_predict_prob = self.inference(dev_output)
        # dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
        #                                              self.dev_class_weight)
        dev_mAP = -1 # utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
        # dev_result = {'output': dev_output.cpu().numpy(), 
        #               'feature': dev_feature.cpu().numpy(),
        #               'per_class_AP': dev_per_class_AP,
        #               'mAP': dev_mAP}
        # utils.save_pkl(dev_result, os.path.join(self.save_path, 'dev_result.pkl'))
        
        test_loss, test_output, test_feature = self._test(self.test_loader)
        test_predict_prob = self.inference(test_output)
        test_per_class_AP = utils.compute_weighted_AP(self.test_target, test_predict_prob, 
                                                     self.test_class_weight)
        test_mAP = utils.compute_mAP(test_per_class_AP, self.subclass_idx)
        test_result = {'output': test_output.cpu().numpy(), 
                      'feature': test_feature.cpu().numpy(),
                      'per_class_AP': test_per_class_AP,
                      'mAP': test_mAP}
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
    


            
