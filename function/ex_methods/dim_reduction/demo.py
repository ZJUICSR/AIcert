import copy
from tqdm import tqdm
import os.path as osp 

'''
    Prerequisites:
        torch/torchvision/matplotlib/sklearn
    
    this is a demo file to the APIs for dimension reduction functions,
    including
        (1)PCA
        (2)T-SNE
        (3)SVM
    this methods can be used to observe so-called 'Latent Separability in Backdoor Learning'(mentioned in
    the paper 'Circumventing Backdoor Defenses That Are Based on Latent Separability')

    code plan
        (1) catch the latent feature.
        (2) dimension reduction.
        (3) draw the illustrations.
'''

import torch
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms








