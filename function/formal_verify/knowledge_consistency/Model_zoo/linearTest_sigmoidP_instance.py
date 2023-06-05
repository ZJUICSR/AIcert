import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

__all__ = ['LinearTester']

class LinearTester(torch.nn.Module):

    def __init__(self, input_size, output_size, layers=3, init_weights=True, gpu_id=None,
                 fix_p=False, affine=False, bn=True, kernel_size=3, padding=1, instance_bn=True, state=0):
        super(LinearTester, self).__init__()
        self.layers = layers
        # size = [C,H,W]
        self.affine = affine
        self.input_size = input_size
        self.output_size = output_size
        self.gpu_id = gpu_id
        self.bn = bn
        self.state = state
        self.nonLinearLayers_p_pre = nn.Parameter(torch.tensor([0.0, 0.0]).cuda(self.gpu_id), requires_grad=(not fix_p))

        self.nonLinearLayers_p = self.get_p()
        self.instance_bn = instance_bn
        # self._make_nonLinearLayers()
        # self._make_linearLayers()
        if init_weights:
            self._initialize_weights()
        # For record
        self.nonLinearLayersRecord = torch.zeros((layers, *self.output_size)).cuda(gpu_id)
        # def _make_linearLayers(self):
        inCh = self.input_size[0]
        outCh = self.output_size[0]
        if self.bn:
            if not self.instance_bn:
                self.linearLayers_bn = nn.BatchNorm2d(inCh, affine=self.affine, track_running_stats=False)
            else:
                self.linearLayers_bn = nn.InstanceNorm2d(inCh, affine=self.affine, track_running_stats=False)
        linearLayers_conv = []
        nonLinearLayers_ReLU = []
        for x in range(self.layers):
            linearLayers_conv += [nn.Conv2d(inCh, outCh, kernel_size=kernel_size, padding=padding, bias=False)]
            nonLinearLayers_ReLU += [nn.ReLU(inplace=True)]
        self.linearLayers_conv = nn.ModuleList(linearLayers_conv)
        self.nonLinearLayers_ReLU = nn.ModuleList(nonLinearLayers_ReLU)

        if not instance_bn:
            self.nonLinearLayers_norm = nn.Parameter(torch.ones(self.layers, self.output_size[0]),
                                                     requires_grad=False)
            self.running_times = nn.Parameter(torch.zeros(self.layers, dtype=torch.long), requires_grad=False)

        else:
            self.nonLinearLayers_norm = torch.ones(self.layers - 1, 1, self.output_size[0]).cuda(self.gpu_id)
            # self.nonLinearLayers_norm = torch.ones(self.layers - 1).cuda(self.gpu_id)

    def get_p(self):
        return nn.Sigmoid()(self.nonLinearLayers_p_pre)

    def forward(self, x):
        self.nonLinearLayers_p = self.get_p()
        if self.bn:
            x = self.linearLayers_bn(x)
        else:
            x = self.my_bn(self.layers - 1, x)

        out = self.linear(self.state, x, torch.zeros_like(x))
        for i in range(1 + self.state, self.layers):
            out = self.nonLinear(i - 1, out)
            out = self.linear(i, x, out)
        return out

    def my_bn(self, i, out, momentum=0.1, eps=1e-5, rec=False, yn=False):
        if not self.instance_bn:
            if self.training:
                a = out.transpose(0, 1).reshape([out.shape[1], -1]).var(-1).sqrt() + eps
                if self.running_times[i] == 0:
                    self.nonLinearLayers_norm[i] = a
                else:
                    self.nonLinearLayers_norm[i] = (1 - momentum) * self.nonLinearLayers_norm[i] + momentum * a
                self.running_times[i] += 1
                a_ = a.reshape(1, out.shape[1], 1, 1)
            else:
                a_ = self.nonLinearLayers_norm[i].reshape(1, out.shape[1], 1, 1)

            a_ = a_.repeat(out.shape[0], 1, out.shape[2], out.shape[3])
            out = out / a_
            return out
        else:
            if not yn:
                a = out.data.reshape([*out.shape[:-2], self.output_size[1]*self.output_size[2]])
                # a = out.data.reshape([*out.shape[:-3],-1]).var(-1).sqrt() \
                #     + eps
                if a.size()[-1] == 1:
                    a = torch.ones_like(a)
                    if rec:
                        self.nonLinearLayers_norm[i] = a.reshape([*a.shape[:-1]])
                else:
                    a = a.var(-1).sqrt() + eps
                    if rec:
                        self.nonLinearLayers_norm[i] = a.squeeze(0)
            else:
                a = self.nonLinearLayers_norm[i]
            a = a.reshape([*out.shape[:-2], 1, 1])
            # a = a.reshape([out.shape[0],1, 1, 1])
            out = out / a
            return out

    def nonLinear(self, i, out, rec=False):
        out = self.my_bn(i, out, rec=rec)
        out = self.nonLinearLayers_ReLU[i](out)
        if rec:
            self.nonLinearLayersRecord[i] = torch.gt(out, 0)#.reshape(self.input_size)
        out = self.nonLinearLayers_p[i] * out
        return out

    def linear(self, i, x, out):
        out = x + out
        out = self.linearLayers_conv[i](out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Useful functions

    def val_linearity(self, x):
        with torch.no_grad():
            bs = x.size(0)
            self.nonLinearLayers_p = self.get_p()
            self.nonLinearLayersRecord = torch.zeros((self.layers, *self.output_size)).to(x.device)
            Yn = torch.zeros((self.layers, *self.output_size))

            if x.size(0) != 1 or self.training:
                print("Only single mode! and Val mode!")
                return
            # record nonLinearLayersRecord
            if self.bn:
                x = self.linearLayers_bn(x)
            else:
                x = self.my_bn(self.layers - 1, x)

            Y_sum = self.linear(0, x, torch.zeros_like(x)).to(x.device)
            for i in range(1, self.layers):
                Y_sum = self.nonLinear(i - 1, Y_sum, True)
                Y_sum = self.linear(i, x, Y_sum)
            Y_sum = Y_sum.reshape(self.output_size).cpu()
            # record Yn
            for n in range(self.layers):
                z = torch.zeros_like(x).cuda(self.gpu_id)
                out = torch.zeros_like(x).cuda(self.gpu_id)
                if n == self.layers - 1:
                    out = self._yn_linear(0, x, out)
                for i in range(1, self.layers):
                    n_ = self.layers - i - 1
                    if n == n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self._yn_linear(i, x, out)
                    elif n > n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self._yn_linear(i, z, out)
                Yn[n] = out.reshape(self.output_size).cpu()
                # Yn_length[n] = torch.sum(Yn[n] ** 2) ** 0.5
            # Yn_contribution = Yn_length ** 2 / torch.sum(Yn_length ** 2)

            return Y_sum, Yn, None, None

    def val_batch(self, x):
        with torch.no_grad():
            bs = x.size(0)
            self.nonLinearLayersRecord = torch.zeros((self.layers, bs, *self.output_size)).to(x.device)
            if self.instance_bn:
                self.nonLinearLayers_norm = torch.ones(self.layers - 1, bs, self.output_size[0]).cuda(self.gpu_id)
            self.nonLinearLayers_p = self.get_p()
            Yn = torch.zeros((self.layers, bs, *self.output_size))
            # record nonLinearLayersRecord
            if self.bn:
                x = self.linearLayers_bn(x)
            else:
                x = self.my_bn(self.layers - 1, x)

            Y_sum = self.linear(0, x, torch.zeros_like(x)).to(x.device)
            for i in range(1, self.layers):
                Y_sum = self.nonLinear(i - 1, Y_sum, True)
                Y_sum = self.linear(i, x, Y_sum)

            # record Yn
            for n in range(self.layers):
                z = torch.zeros_like(x).to(x.device)
                out = torch.zeros_like(x).to(x.device)
                if n == self.layers - 1:
                    out = self._yn_linear(0, x, out)
                for i in range(1, self.layers):
                    n_ = self.layers - i - 1
                    if n == n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self._yn_linear(i, x, out)
                    elif n > n_:
                        out = self._yn_nonLinear(i - 1, out)
                        out = self._yn_linear(i, z, out)
                Yn[n] = out

            return Y_sum, Yn


    def _yn_nonLinear(self, i, out):
        out = self.my_bn(i, out, yn=True)
        out = self.nonLinearLayersRecord[i] * out
        out = self.nonLinearLayers_p[i] * out
        return out

    def _yn_linear(self, i, x, out):
        out = x + out
        out = self.linearLayers_conv[i](out)
        return out

def main():
    model = LinearTester((3,5,5),(3,5,5),bn=False, gpu_id=0).cuda(0)
    model.eval()
    z = torch.randn(4,3,5,5).cuda()
    x,y = model.val_batch(z)
    print(x.shape,y.shape)

if __name__ == '__main__':
    main()
