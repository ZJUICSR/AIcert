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

    def __init__(self, input_channel):
        super(LeNet, self).__init__()
        self.feature = [
            Conv2d(in_channels=input_channel, out_channels=6, kernel_size=5),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            ]
        self.classifer = [
            Linear(in_features=16 * 5 * 5, out_features=120),
            ReLU(),
            Linear(in_features=120, out_features=84),
            ReLU(),
            Linear(in_features=84, out_features=10, whichScore=None, lastLayer=True)
        ]

    def forward(self):
        layers = self.feature + self.classifer
        return Sequential(*layers)


def lenet(reference_model, input_channel):
    model = LeNet(input_channel).forward()
    if reference_model != None:
        model_keys = list(reference_model.state_dict().keys())
        net_keys = list(model.state_dict().keys())

        for i in range(len(model_keys)):
            try:
                model.state_dict()[net_keys[i]][:] = reference_model.state_dict()[model_keys[i]][:]
            except:
                model.state_dict()[net_keys[i]] = reference_model.state_dict()[model_keys[i]]
    return model
