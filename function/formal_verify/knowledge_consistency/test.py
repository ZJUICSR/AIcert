import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.cuda as cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from logger import Logger
import os
from torch.nn.modules.loss import _Loss
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import PIL.Image as Image
import Model_zoo as models
import matplotlib.colors as colors
import cv2
denormalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225])
toPIL = T.Compose([denormalize, T.ToPILImage()])

device_ids = [1,0]
gpu = 1
model_path1 = "./model_checkpoints/checkpoint_CUB200_vgg16_bn_lr-2_sd0_itr300.pth.tar"
model_path2 = "./model_checkpoints/checkpoint_CUB200_vgg16_bn_lr-2_sd5_itr300.pth.tar"
image_path = "../image/"
resume = "./model_checkpoints/checkpoint_L30__a0.1_lr-4.pth.tar"
dataset = 'CUB200'
conv_layer = 30
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
arch = 'vgg16_bn'

def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(device_ids[0])))
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module') >= 0:
                state_dict[key.replace('module.','')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
              .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

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

def get_feature(img, net):
    input = img.unsqueeze(0)
    net.eval()
    with torch.no_grad():
        if arch.startswith("alexnet"):
            x = net.features[:conv_layer + 1](input)

        elif arch.startswith("vgg"):
            x = net.features[:conv_layer + 1](input)
            print(x.shape)


        elif arch.startswith("resnet"):
            x = input
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)

            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3[:-1](x)
            x = ResBlock_beforeReLU(net.layer3[-1], x)

        return x.squeeze()
def vis(y, t, i, h=3,w=4,universal = True):
    cmap = 'jet'
    fig1, axes = plt.subplots(h,w,figsize=(10,10))

    im0 = axes[0][0].imshow(toPIL(image_dataset[i][0]))
    for k in range(1,h*w):
        if universal:
            im1 = axes[k//w][k%w].imshow(y[t[k-1]],cmap=cmap,vmax=y.max(), vmin=y.min())
        else:
            im1 = axes[k//w][k%w].imshow(y[t[k-1]],cmap=cmap)
    fig1.tight_layout()
    fig1.subplots_adjust(right=0.8)
    cbar_ax2 = fig1.add_axes([0.85, 0.10, 0.05, 0.80])
    fig1.colorbar(im1, cax=cbar_ax2)
    #plt.show()
    plt.savefig("res.png")
    plt.close()

net1 = models.__dict__[arch](num_classes=200)
load_checkpoint(model_path1, net1)
net2 = models.__dict__[arch](num_classes=200)
load_checkpoint(model_path2, net2)
print(net1)

traindir = "./image"
image_dataset = datasets.ImageFolder(
        traindir, T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            normalize,
        ]))

x = get_feature(image_dataset[0][0], net1)

input_size = x.shape
output_size = x.shape

print (input_size)

model = models.LinearTester(input_size,output_size, gpu_id= gpu, affine=False, bn = False, instance_bn=True).cuda(gpu)
checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(gpu)))
model.load_state_dict(checkpoint['state_dict'])
del checkpoint

bird = 0 
input = get_feature(image_dataset[bird][0],net1)
target = get_feature(image_dataset[bird][0],net2)
print(input.shape)
vis(input, np.arange(35)*2, bird,6,6,1)
vis(target, np.arange(35)*2, bird,6,6,1)
input = input.unsqueeze(0).cuda(gpu)
model.eval()
output, output_n, output_contrib, res = model.val_linearity(input)
vis(output, np.arange(35)*2, bird,6,6,1)