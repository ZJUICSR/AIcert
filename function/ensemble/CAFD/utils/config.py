import numpy as np
import math
import torch
from torchvision import transforms

mean=np.array([0.4914, 0.4822, 0.4465])
std=np.array([0.2023, 0.1994, 0.2010])

max_val = np.array([ (1. - mean[0]) / std[0],
                     (1. - mean[1]) / std[1],
                     (1. - mean[2]) / std[2],
                    ])

min_val = np.array([ (0. - mean[0]) / std[0],
                     (0. - mean[1]) / std[1],
                     (0. - mean[2]) / std[2],
                   ])
                    

eps_size=np.array([  abs( (1. - mean[0]) / std[0] ) + abs( (0. - mean[0]) / std[0] ),
                     abs( (1. - mean[1]) / std[1] ) + abs( (0. - mean[1]) / std[1] ),
                     abs( (1. - mean[2]) / std[2] ) + abs( (0. - mean[2]) / std[2] ),
                  ])



def train_scale():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def train_zero_norm():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def test_scale():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def test_zero_norm():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def unnormalize():
    return transforms.Normalize( (- mean / std).tolist(), (1.0 / std ).tolist() )

def inverse_normalize():
    u = [-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010]
    sigma = [1./0.2023, 1./0.1994, 1./0.2010]
    return transforms.Normalize( u, sigma )



def compute_lr(lr, itr):
    if itr < 75:
        return lr
    else:
        return 0.01
    return lr * math.pow(0.2, optim_factor)


# def compute_lr(lr, itr):
#     optim_factor = 0
#     if itr > 80:
#         optim_factor = 3
#     elif itr > 60:
#         optim_factor = 2
#     elif itr > 30:
#         optim_factor = 1

#     return lr * math.pow(0.2, optim_factor)


def accuracy(output, y, k=1):
    """Computes the precision@k for the specified values of k"""
    # Rehape to [N, 1]
    target = y.view(-1, 1)

    _, pred = torch.topk(output, k, dim=1, largest=True, sorted=True)
    correct = torch.eq(pred, target)

    return torch.sum(correct).float() / y.size(0)



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


# mean = {
#     'cifar10': (0.4914, 0.4822, 0.4465),
#     'cifar100': (0.5071, 0.4867, 0.4408),
# }

# std = {
#     'cifar10': (0.2023, 0.1994, 0.2010),
#     'cifar100': (0.2675, 0.2565, 0.2761),
# }