import torch
from .sequential import Sequential
from .linear import Linear
from .relu import ReLU
from .module import Module
from .convolution import Conv2d
from .pool import MaxPool2d


class LeNet(Module):
    """
        使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
        """

    def __init__(self):
        super(LeNet, self).__init__()
        self.output_size = 3
        self.feature = Sequential(
            Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1),
            ReLU(),
            Linear(in_features=24 * self.output_size * self.output_size, out_features=120),
            ReLU(),
            Linear(in_features=120, out_features=60),
            ReLU(),
            Linear(in_features=60, out_features=10, whichScore=None, lastLayer=True)
        )

    def forward(self, x):
        x = self.feature(x)
        return x


def lenet(pretrained=False, reference_model=None):
    net = torch.load("models/mnist_cnn.pt")
    if pretrained == True:
        return net.feature
    else:
        return LeNet().feature
