"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from .resnet import BasicBlock


class ParsevalBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        # convex combination here
        out = 0.5 * (self.residual_function(x) + self.shortcut(x))
        return nn.ReLU(inplace=True)(out)

    def __iter__(self):
        return iter(self.residual_function)


class ParsevalResNet(nn.Module):
    # record current blocks
    current_block: int = 0

    def __init__(self, k: int, num_block, num_classes=100):
        """
        Args:
            k: the last k blocks which will be retrained
        """
        super().__init__()

        self._k = k
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(64, num_block[0], 1)
        self.conv3_x = self._make_layer(128, num_block[1], 2)
        self.conv4_x = self._make_layer(256, num_block[2], 2)
        self.conv5_x = self._make_layer(512, num_block[3], 2)

        type(self).reset_current_block()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            type(self).current_block += 1
            # use ParsevalBasicBlock for residual block that needs retraining
            # 9 is the total block of resnet18
            if type(self).current_block + self._k > 9:
                block = ParsevalBasicBlock
            else:
                block = BasicBlock
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    @classmethod
    def reset_current_block(cls):
        cls.current_block = 0


def parseval_resnet18(k: int, num_classes: int = 100):
    """
    Args:
        k: the last k blocks which will be retrained
    """
    return ParsevalResNet(k, [2, 2, 2, 2], num_classes=num_classes)
