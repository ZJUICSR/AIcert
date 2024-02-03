import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=False):
        super(wide_basic, self).__init__()
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        if dropout_rate:
            self.dropout = nn.Dropout(p=dropout_rate)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        # out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        if self.use_bn:
            out = self.conv1(F.relu(self.bn1(x)))
        else:
            out = self.conv1(F.relu(x))
        if self.dropout_rate:
            out = self.dropout(out)
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv2(F.relu(out))

        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, use_bn=False, use_pooling=True):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.use_bn = use_bn
        self.use_pooling = use_pooling
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [self.in_planes, self.in_planes*2*k, self.in_planes*4*k, self.in_planes*8*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        # self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.1)
        if self.use_pooling:
            self.linear1 = nn.Linear(nStages[3], 512)
        else:
            self.linear1 = nn.Linear(nStages[3]*64, 512)

        self.linear2 = nn.Linear(512, num_classes)


    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, self.use_bn))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        if self.use_pooling:
            out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out

def wide_resnet_cifar(in_ch=3, in_dim=32):
    return Wide_ResNet(16, 4, 0.3, 10)

def wide_resnet_cifar_bn(in_ch=3, in_dim=32):
    return Wide_ResNet(10, 4, None, 10, use_bn=True)

def wide_resnet_cifar_bn_wo_pooling(in_ch=3, in_dim=32): # 1113M, 21M
    return Wide_ResNet(10, 4, None, 10, use_bn=True, use_pooling=False)

def wide_resnet_cifar_bn_wo_pooling_dropout(in_ch=3, in_dim=32): # 1113M, 21M
    return Wide_ResNet(10, 4, 0.3, 10, use_bn=True, use_pooling=False)

if __name__ == '__main__':
    from thop import profile
    net = wide_resnet_cifar_bn_wo_pooling_dropout()
    print(net)
    y = net(torch.randn(1,3,32,32))
    macs, params = profile(net, (torch.randn(1, 3, 32, 32),))
    print(macs/1000000, params/1000000)  # 1096M, 5M
    print(y.size())