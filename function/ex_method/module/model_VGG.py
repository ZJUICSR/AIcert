import torch
import torchvision

from .sequential import Sequential
from .linear import Linear
from .relu import ReLU
from .module import Module
from .convolution import Conv2d
from .batchnorm import BatchNorm2d
from .pool import MaxPool2d
import os


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class VGG(Module):
    def __init__(self):
        super(VGG, self).__init__()

    def forward(self, name='E', pretrained=False, reference_model=None, batch_norm=False, whichScore=None, num_classes=1000,output_size=7):
        layers = self.make_layers(cfg[name])

        layers = layers + [Linear(512 * output_size * output_size, 4096),
                           ReLU(),
                           Linear(4096, 4096),
                           ReLU(),
                           Linear(4096, num_classes, whichScore=whichScore, lastLayer=True)]

        if pretrained == False:
            return Sequential(*layers)

        net = Sequential(*layers)

        model = None
        if reference_model == None:
            if batch_norm == True:
                if name == "A":
                    model = torchvision.models.vgg11_bn(pretrained=True)
                elif name == "B":
                    model = torchvision.models.vgg19_bn(pretrained=True)
                elif name == "D":
                    model = torchvision.models.vgg19_bn(pretrained=True)
                elif name == "E":
                    model = torchvision.models.vgg19_bn(pretrained=True)
            else:
                if name == "A":
                    model = torchvision.models.vgg11(pretrained=True)
                elif name == "B":
                    model = torchvision.models.vgg13(pretrained=True)
                elif name == "D":
                    model = torchvision.models.vgg16(pretrained=True)
                elif name == "E":
                    model = torchvision.models.vgg19(pretrained=True)
        else:
            model = reference_model

        model_keys = list(model.state_dict().keys())
        net_keys = list(net.state_dict().keys())

        for i in range(len(model_keys)):
            try:
                net.state_dict()[net_keys[i]][:] = model.state_dict()[model_keys[i]][:]
            except:
                net.state_dict()[net_keys[i]] = model.state_dict()[model_keys[i]]
        return net

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, BatchNorm2d(v), ReLU()]
                else:
                    layers += [conv2d, ReLU()]
                in_channels = v
        return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
