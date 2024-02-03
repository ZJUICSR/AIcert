# Modified  from https://github.com/xinntao/BasicSR
import functools
import torch
import torch.nn as nn
import function.ensemble.CAFD.modules.module_util as mutil



###############################################################
# NRP network
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class NRP(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(NRP, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))

        return trunk


#################################################################
# NRP based on ResNet Generator
class NRP_resG(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23):
        super(NRP_resG, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = mutil.make_layer(basic_block, nb)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.conv_last(self.recon_trunk(fea))
        return out


class Discriminator(nn.Module):# VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.
    It is used to train SRGAN and ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(Discriminator, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            num_feat * 2, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 1, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        # self.linear1 = nn.Linear(num_feat * 4 * 2 * 2, 100)
        # self.linear1 = nn.Linear(10368, 100)
        if num_in_ch == 1:
            self.linear1 = nn.Linear(100352, 100)
        else:
            self.linear1 = nn.Linear(131072, 100)
        '''
        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        
        self.conv4_0 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        
        self.conv4_0 = nn.Conv2d(
            num_feat * 8, num_feat * 16, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 16, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_feat * 16, num_feat * 16, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 16, affine=True)

        self.conv5_0 = nn.Conv2d(
            num_feat * 16, num_feat * 16, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(num_feat * 16, affine=True)
        self.conv5_1 = nn.Conv2d(
            num_feat * 16, num_feat * 16, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(num_feat * 16, affine=True)
        
        self.linear1 = nn.Linear(num_feat * 16 * 4 * 4, 100)
        '''
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # assert x.size(2) == 32 and x.size(3) == 32, (
        #     f'Input spatial size must be 128x128, '
        #     f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (128, 128) (16,16)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (64, 64) (8, 8)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (32, 32) (4, 4)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (16, 16) (2, 2)
        '''
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (8, 8)
        
        feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
        feat = self.lrelu(self.bn5_1(
            self.conv5_1(feat)))  # output spatial size: (4, 4)
        '''
        feat = feat.view(feat.size(0), -1)
        # print(f'featxxxxxx={feat.shape}')
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

class DiscriminatorCifar(nn.Module):  # VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.
    It is used to train SRGAN and ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(DiscriminatorCifar, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.linear1 = nn.Linear(num_feat * 4 * 2 * 2, 100)
        # self.linear1 = nn.Linear(10368, 100)
        '''
        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)

        self.conv4_0 = nn.Conv2d(
            num_feat * 8, num_feat * 16, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 16, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_feat * 16, num_feat * 16, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 16, affine=True)

        self.conv5_0 = nn.Conv2d(
            num_feat * 16, num_feat * 16, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(num_feat * 16, affine=True)
        self.conv5_1 = nn.Conv2d(
            num_feat * 16, num_feat * 16, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(num_feat * 16, affine=True)

        self.linear1 = nn.Linear(num_feat * 16 * 4 * 4, 100)
        '''
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # assert x.size(2) == 32 and x.size(3) == 32, (
        #     f'Input spatial size must be 128x128, '
        #     f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (128, 128) (16,16)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (64, 64) (8, 8)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (32, 32) (4, 4)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (16, 16) (2, 2)
        '''
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
        feat = self.lrelu(self.bn5_1(
            self.conv5_1(feat)))  # output spatial size: (4, 4)
        '''
        feat = feat.view(feat.size(0), -1)
        # print(f'featXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX={feat.shape}')
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


from torchvision.models.densenet import densenet121
if __name__ == '__main__':
    # netG = NRP_resG(3, 3, 64, 23)
    # netG.load_state_dict(torch.load('pretrained_purifiers/NRP_resG.pth'))
    # test_sample = torch.rand(1, 3, 256, 256)
    # print(netG(test_sample).size())
    # #print(netG(test_sample).size())
    # print(sum(p.numel() for p in netG.parameters() if p.requires_grad))
    model = Discriminator(3, 32)
    data = torch.randn([20, 3, 32, 32])
    print(model(data).shape)