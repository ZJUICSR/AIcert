import numpy as np
import math
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

import sys


class Preprocessor:
    """ We Assyne CIFAR-10"""
    def __init__(self, ds_name, ds_path='datasets/', normalized=None):
        self.train_data = None
        self.test_data = None
        self.max_val = None
        self.min_val = None
        self.eps_size = None
        
        # Retrieve plain dataset
        train_data, test_data = self.get_dataset(ds_path, ds_name, self.test_scale, self.test_scale)
        
        # Compute mean and std
        self.mean, self.std = self.compute_stats(train_data)
        
        # Retrieve augmented dataset
        if normalized:
            
            print(">>> NORMALIZING IMAGES WITH ZERO-MEAN...")

            self.train_data, self.test_data = self.get_dataset(ds_path, ds_name, 
                                                               self.train_normalize,
                                                               self.test_normalize)

            
            self.max_val, self.min_val, self.eps_size = self.init_limits(self.mean, self.std)
        else:
            print(">>> SCALING IMAGES [0-1]...")
            self.train_data, self.test_data = train_data, test_data
        
        print(">>> Dataset Mean:", self.mean.data)
        print(">>> Dataset Std:", self.std.data)
            
        
        
    def compute_stats(self, train_data):
        """ Returns the mean and std."""
        mean_mat = torch.stack([torch.mean(t, dim=(1,2)) for t, c in train_data])
        mean = torch.mean(mean_mat, dim=0)
        
        std_mat = torch.stack([ torch.std(img, dim=(1,2)) for img, _ in train_data])
        std = torch.mean(std_mat, dim=0)
        
        return mean, std

    
    def get_dataset(self, ds_path, ds_name, train_transform, test_transform):
        """ Returns torchvision Dataset Object based argument."""
        train_data, test_data = None, None
        if ds_name == 'CIFAR10':
            train_data = CIFAR10(ds_path, train=True, transform=train_transform(), download=True)
            test_data = CIFAR10(ds_path, train=False, transform=test_transform(), download=True)
            
        return train_data, test_data
    
    
    def init_limits(self, mean, std):
        """ Returns the max values allowed for normalized images."""
        max_val = np.zeros_like(mean)
        min_val = np.zeros_like(mean)
        eps_size = np.zeros_like(mean)
        for i in range(len(mean)):
            max_val[i] = (1. - mean[i]) / std[i]
            min_val[i] = (0. - mean[i]) / std[i]
            eps_size[i] = abs( (1. - mean[i]) / std[i] ) + abs( (0. - mean[i]) / std[i] )
            
        return max_val, min_val, eps_size
    
    
    def datasets(self):
        return (self.train_data, self.test_data, len(self.train_data), len(self.test_data))
    
    
    def get_const(self):
        return (self.max_val, self.min_val, self.eps_size)
    
    
    def train_scale(self):
        """ Scale to [0-1] on Training Set"""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def test_scale(self):
        """ Scale to [0-1] on Test Set"""
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    

    def train_normalize(self):
        """ Per-Channel Zero-Mean Normalization on Train set. """
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def test_normalize(self):
        """ Per-Channel Zero-Mean Normalization on Test set. """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
    
    
    def inverse_normalize():
        """ Undo Zero-Mean Normalization"""
        return transforms.Normalize( (- self.mean / self.std).tolist(), (1.0 / self.std ).tolist() )
    
    
    
# ======================================= AVERAGE CALCULATOR ========================================

class AverageMeter():
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

        
def accuracy(logits, y, K=1):
    """Computes the precision@k for the specified values of k"""
    # Rehape to [N, 1]
    target = y.view(-1, 1)

    _, pred = torch.topk(logits, K, dim=1, largest=True, sorted=True)
    correct = torch.eq(pred, target)

    return torch.sum(correct).float() / y.size(0)