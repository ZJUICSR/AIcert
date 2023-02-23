import torch
import torchvision
from .sequential import Sequential
from .linear import Linear
from .relu import ReLU
from .module import Module
from .convolution import Conv2d
from .pool import MaxPool2d


class Network_mnist(Module):
    """
        使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
        """

    def __init__(self, parameters):
        super(Network_mnist, self).__init__()
        self.output_size = 3
        self.feature = Sequential(
            Conv2d(in_channels=1, out_channels=6, kernel_size=parameters["conv_k"], stride=parameters["conv_stride"], padding=parameters["conv_padding"]),
            ReLU(),
            MaxPool2d(kernel_size=parameters["pool_k"]),
            Conv2d(in_channels=6, out_channels=12, kernel_size=parameters["conv_k"], stride=parameters["conv_stride"], padding=parameters["conv_padding"]),
            ReLU(),
            MaxPool2d(kernel_size=parameters["pool_k"]),
            Conv2d(in_channels=12, out_channels=24, kernel_size=parameters["conv_k"], stride=parameters["conv_stride"], padding=parameters["conv_padding"]),
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


def mnist_self(pretrained=False, reference_model=None):
    checkpoint = torch.load("models/mnist_model.pkl")
    parameters = checkpoint["parameters"]
    if pretrained == False:
        net = Network_mnist(parameters)
        return net.feature
    else:
        net = checkpoint["model"]
    return net.feature
