import torch
import torch.nn as nn
import torch.nn.functional as F

from .wrn import BasicBlock


class ParsevalBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(ParsevalBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        # todo
        # convex combination
        former_out = (x if self.equalInOut else self.convShortcut(x)) * 0.5
        out = out * 0.5

        return torch.add(former_out, out)
        # return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def __iter__(self):
        return iter(
            [self.bn1, self.relu1, self.conv1, self.bn2, self.relu2, self.conv2]
        )


class ParsevalNetworkBlock(nn.Module):
    # record current residual block
    current_block: int = 0
    total_blocks: int = 0

    def __init__(self, k: int, nb_layers, in_planes, out_planes, stride, dropRate=0.0):
        """
        Args:
            k: the last k blocks which will be retrained
        """
        super(ParsevalNetworkBlock, self).__init__()
        self.layer = self._make_layer(k, in_planes, out_planes, nb_layers, stride, dropRate)

    @classmethod
    def _make_layer(cls, k: int, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            cls.current_block += 1
            if cls.current_block + k > cls.total_blocks:
                block = ParsevalBasicBlock
            else:
                block = BasicBlock
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

    @classmethod
    def set_total_blocks(cls, n_tb):
        cls.total_blocks = n_tb

    @classmethod
    def reset_current_block(cls):
        cls.current_block = 0


class ParsevalWideResNet(nn.Module):
    def __init__(self, k: int, depth, num_classes, widen_factor=1, dropRate=0.0):
        """wide resnet for parseval training

        Args:
            k: the last k blocks which will be retrained
        """
        super(ParsevalWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # set number of total blocks
        ParsevalNetworkBlock.set_total_blocks(2 + n * 3)
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = ParsevalNetworkBlock(k, n, nChannels[0], nChannels[1], 1, dropRate)
        # 2nd block
        self.block2 = ParsevalNetworkBlock(k, n, nChannels[1], nChannels[2], 2, dropRate)
        # 3rd block
        self.block3 = ParsevalNetworkBlock(k, n, nChannels[2], nChannels[3], 2, dropRate)
        ParsevalNetworkBlock.reset_current_block()
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def parseval_retrain_wrn34_10(k: int, num_classes=10):
    """
    Args:
        k: the last k blocks which will be retrained
    """
    return ParsevalWideResNet(k, 34, num_classes, 10, 0)

def parseval_retrain_wrn28_10(k: int, num_classes=10):
    return ParsevalWideResNet(k, 28, num_classes, 10, 0)


def parseval_retrain_wrn28_4(k: int, num_classes=10):
    return ParsevalWideResNet(k, 28, num_classes, 4, 0)


def parseval_normal_wrn34_10(num_classes=10):
    return ParsevalWideResNet(17, 34, num_classes, 10, 0)
