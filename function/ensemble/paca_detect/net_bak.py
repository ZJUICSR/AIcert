import torch.nn as nn
import torch.nn.functional as F
from .padding_same_conv import Conv2d
import torch.nn.init as init
from .MPNCOV import MPNCOV, CovpoolLayer, SqrtmLayer, TriuvecLayer


class SRNcovent(nn.Module):
    def __init__(self, with_bn=False, threshold=3, channel=3):
        super(SRNcovent, self).__init__()

        self.layer1 = Conv2d(in_channels=channel, out_channels=64, kernel_size=5, stride=1, dilation=1, groups=1, bias=True)
        self.layer1_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.05, affine=True,
                                        track_running_stats=False)

        self.layer2 = Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1, bias=True)
        self.layer2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                        track_running_stats=False)

        self.layer3_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer3_1_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer3_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer3_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)

        self.layer4_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer4_1_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer4_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer4_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)

        self.layer5_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer5_1_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer5_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer5_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)

        self.layer6_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer6_1_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer6_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer6_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)

        self.layer7_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer7_1_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer7_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer7_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)

        self.layer8_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, dilation=1, groups=1,
                               bias=True)
        self.layer8_1_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer8_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer8_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer8_2_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=1, groups=1,
                                 bias=True)
        self.layer8_2_2_bn = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.05, affine=True,
                                            track_running_stats=False)

        self.layer9_1 = Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=2, dilation=1, groups=1,
                               bias=True)
        self.layer9_1_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer9_2 = Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, dilation=1, groups=1,
                               bias=True)
        self.layer9_2_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.05, affine=True,
                                          track_running_stats=False)
        self.layer9_2_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, groups=1,
                                 bias=True)
        self.layer9_2_2_bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.05, affine=True,
                                            track_running_stats=False)

        self.layer10_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, dilation=1, groups=1,
                                bias=True)
        self.layer10_1_bn = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.05, affine=True,
                                           track_running_stats=False)
        self.layer10_2 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, dilation=1, groups=1,
                                bias=True)
        self.layer10_2_bn = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.05, affine=True,
                                           track_running_stats=False)
        self.layer10_2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1, groups=1,
                                  bias=True)
        self.layer10_2_2_bn = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.05, affine=True,
                                             track_running_stats=False)

        self.layer11_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, dilation=1, groups=1,
                                bias=True)
        self.layer11_1_bn = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.05, affine=True,
                                           track_running_stats=False)
        self.layer11_2 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, dilation=1, groups=1,
                                bias=True)
        self.layer11_2_bn = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.05, affine=True,
                                           track_running_stats=False)
        self.layer11_2_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, groups=1,
                                  bias=True)
        self.layer11_2_2_bn = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.05, affine=True,
                                             track_running_stats=False)

        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)

        init.xavier_normal(self.fc1.weight)
        init.constant(self.fc1.bias, 0)

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer1_bn(x)

        x = self.layer2(x)
        x = self.layer2_bn(x)

        x = self.layer3_2_bn(self.layer3_2(F.relu(self.layer3_1_bn(self.layer3_1(x)))))+x

        x = self.layer4_2_bn(self.layer4_2(F.relu(self.layer4_1_bn(self.layer4_1(x)))))+x 

        x = self.layer5_2_bn(self.layer5_2(F.relu(self.layer5_1_bn(self.layer5_1(x)))))+x

        x = self.layer6_2_bn(self.layer6_2(F.relu(self.layer6_1_bn(self.layer6_1(x)))))+x

        x = self.layer7_2_bn(self.layer7_2(F.relu(self.layer7_1_bn(self.layer7_1(x)))))+x

        x = self.layer8_1_bn(self.layer8_1(x))+F.avg_pool2d(self.layer8_2_2_bn(self.layer8_2_2(F.relu(self.layer8_2_bn(self.layer8_2(x))))),3,2,padding=1)

        x = self.layer9_1_bn(self.layer9_1(x))+F.avg_pool2d(self.layer9_2_2_bn(self.layer9_2_2(F.relu(self.layer9_2_bn(self.layer9_2(x))))),3,2,padding=1)

        x = self.layer10_1_bn(self.layer10_1(x))+F.avg_pool2d(self.layer10_2_2_bn(self.layer10_2_2(F.relu(self.layer10_2_bn(self.layer10_2(x))))),3,2,padding=1)

        x = self.layer11_1_bn(self.layer11_1(x))+F.avg_pool2d(self.layer11_2_2_bn(self.layer11_2_2(F.relu(self.layer11_2_bn(self.layer11_2(x))))),3,2,padding=1)

        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class TwoStraeamSrncovet(nn.Module):
    def __init__(self, channel=3):
        super(TwoStraeamSrncovet, self).__init__()
        self.spatial = SRNcovent(channel=channel)
        self.gradient = SRNcovent(channel=channel)

    def forward(self, x, grad):
        logit1 = self.spatial(x)
        logit2 = self.gradient(grad)
        final_logit = logit1 + logit2
        return final_logit


if __name__ == '__main__':
    import torch
    net = TwoStraeamSrncovet()
    torch.save(net.state_dict(), f'twostreamnet.pkl')
