'''
This is a collection of functions used fo network compressing!

'''
import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
#from logger import Logger
from torch.utils.data.sampler import SubsetRandomSampler
#from convOut_loader import convOut_Dataset
#from linearTest_sigmoidP_instance import LinearTester
#from NewVOCDataset import NewVOCDataset
import Model_zoo as models
import threading
import copy
import os
from tqdm import tqdm
from torch.nn.modules.loss import _Loss
#from DiscreteDataset_ABCDE import DiscreteDataset_ABCDE
from tqdm import tqdm
import copy
import math
import numpy as np
import torch
from torch.nn import Parameter,init
from torch.nn.modules.module import Module

import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module') != -1:
                state_dict[key.replace('module.','')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        if 'best_acc1' in checkpoint:
            print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
              .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    del checkpoint, state_dict
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count += n
        self.avg = self.sum / self.count
def ResBlock_beforeReLU(block, x): # Only for ResNet 50 101 152!!
    identity = x

    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)

    out = block.conv2(out)
    out = block.bn2(out)
    if block.__class__.__name__ != 'BasicBlock':
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(x)

    out += identity
    return out
def conv_feature(input, model, conv_layer, arch, device_id):
    input = input.cuda(device_id, non_blocking=True)
    model.eval()
    with torch.no_grad():
        if arch.startswith("alexnet"):
            x = model.features[:conv_layer + 1](input)

        elif arch.startswith("vgg"):
            x = model.features[:conv_layer + 1](input)
        
        elif arch.startswith("resnet"):
            x = input
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3[:-1](x)
            x = ResBlock_beforeReLU(model.layer3[-1], x)        
    return x

class MaskedConv2d(Module): # for conv layer
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = (0,0), padding = (0,0)):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        # Mask works only on weight but not on bias, we first initialize mask to one
        self.mask = Parameter(torch.ones([out_channels, in_channels,*kernel_size]), requires_grad=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
            
            
    def forward(self, input):
        return F.conv2d(input, self.weight*self.mask, self.bias, self.stride,
                        self.padding)

    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
        
def mask_fn(model):
    # change the FC layers into masked layers
    # TODO: make it more adorable
    fc1 = MaskedLinear(25088,4096)
    fc1.weight.data = model.classifier[0].weight.data
    fc1.bias.data = model.classifier[0].bias.data
    model.classifier[0] = fc1
    
    fc2 = MaskedLinear(4096,4096)
    fc2.weight.data = model.classifier[3].weight.data
    fc2.bias.data = model.classifier[3].bias.data
    model.classifier[3] = fc2
    
    fc3 = MaskedLinear(4096,200)
    fc3.weight.data = model.classifier[6].weight.data
    fc3.bias.data = model.classifier[6].bias.data
    model.classifier[6] = fc3
    
def mask_conv(model):
    # change the convolution layers into masked layers
    for name, module in model.named_modules():
        if(isinstance(module,nn.Conv2d)):
            # print(module)
            dev_conv = MaskedConv2d(module.in_channels,module.out_channels,module.kernel_size,module.stride,module.padding)
            dev_conv.weight.data = module.weight.data
            dev_conv.bias.data = module.bias.data
            model.features[int(name[9:])] = dev_conv


def load_masked_model_as_original_model(resume = "vgg16_bn_ft_s_fc_1.00_s_conv_0.20_retrain_centroid_epoch_20_acc_61.22_lr-2_sd44.pth.tar"):
    model = models.__dict__['vgg16_bn_ft'](num_classes=200)

    mask_fn(model)
    mask_conv(model)
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location=torch.device("cuda:0"))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}'".format(resume))
    
    # change masked FC layers into simple linear layers
    # TODO: make it more adorable
    fc1 = nn.Linear(25088,4096)
    fc1.weight.data = model.classifier[0].weight.data
    fc1.bias.data = model.classifier[0].bias.data
    model.classifier[0] = fc1

    fc2 = nn.Linear(4096,4096)
    fc2.weight.data = model.classifier[3].weight.data
    fc2.bias.data = model.classifier[3].bias.data
    model.classifier[3] = fc2

    fc3 = nn.Linear(4096,200)
    fc3.weight.data = model.classifier[6].weight.data
    fc3.bias.data = model.classifier[6].bias.data
    model.classifier[6] = fc3
                
    for name, module in model.named_modules():
        if(isinstance(module,MaskedConv2d)):
            # print(module)
            dev_conv = nn.Conv2d(module.in_channels,module.out_channels,module.kernel_size,module.stride,module.padding)
            dev_conv.weight.data = module.weight.data
            dev_conv.bias.data = module.bias.data
            model.features[int(name[9:])] = dev_conv
    
    return model