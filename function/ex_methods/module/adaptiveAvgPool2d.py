import torch

from.module import Module
from torch.nn import functional as F
from ._jit_internal import weak_module, weak_script_method


@weak_module
class _AvgPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad']

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )

@weak_module
class AvgPool2d(_AvgPoolNd):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    @weak_script_method
    def forward(self, input):
        self.input = input
        self.output = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        return self.output

    def get_grad_and_activation(self, grad_output, requires_activation):
        '''
        Note that this implementation is only for the global average pooling. 
        If the output size of this layer is not 1, than this implimentation does not work.
        '''
        grad_output = F.interpolate(grad_output, scale_factor=self.kernel_size, mode='nearest')
        grad_input = grad_output / (self.kernel_size**2)

        return grad_input, self.output
    
    def _simple_lrp(self, R, labels, device):
        RdivZ = R/(self.output+1e-8)/ (self.kernel_size**2)
        RdivZ_interpol = F.interpolate(RdivZ, scale_factor=self.kernel_size, mode='nearest')
        
        return self.input*RdivZ_interpol

    def _epsilon_lrp(self, R, epsilon, device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R, device=device)

    def _alphabeta_lrp(self, R, labels, device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R, labels,device)

    def _composite_lrp(self, R, labels, device, beta=0):
        
        x_p = torch.where(self.input<0, torch.zeros(1).to(device), self.input)
        x_n = self.input - x_p
        
        output_p = F.avg_pool2d(x_p, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        output_n = F.avg_pool2d(x_n, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        
        RdivZp = R/(output_p+1e-8)/ (self.kernel_size**2)
        RdivZn = R/(output_n+1e-8)/ (self.kernel_size**2)
        
        RdivZp_interpol = F.interpolate(RdivZp, scale_factor=self.kernel_size, mode='nearest')
        RdivZn_interpol = F.interpolate(RdivZn, scale_factor=self.kernel_size, mode='nearest')

        return (1-beta) * x_p*RdivZp_interpol + beta * x_n*RdivZn_interpol




@weak_module
class _AdaptiveAvgPoolNd(Module):
    __constants__ = ['output_size']

    def __init__(self, output_size):
        super(_AdaptiveAvgPoolNd, self).__init__()
        self.output_size = output_size

    def extra_repr(self):
        return 'output_size={}'.format(self.output_size)




@weak_module
class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    @weak_script_method
    def forward(self, input):
        self.input = input
        self.output = F.adaptive_avg_pool2d(input, self.output_size)
        return self.output

    #190320_added
    
    def get_grad_and_activation(self, grad_output, requires_activation):
        '''
        Note that this implementation is only for the global average pooling. 
        If the output size of this layer is not 1, than this implimentation does not work.
        '''
        scale = self.input.shape[2]
        n,f = grad_output.shape
        grad_output = F.interpolate(grad_output.reshape(n,f,1,1), scale_factor=scale, mode='nearest')
        grad_input = grad_output / (self.input.shape[2]*self.input.shape[3])

        return grad_input, None
    
    def _simple_lrp(self, R, labels,device):
        R = R.unsqueeze(-1).unsqueeze(-1)
        lrp_input = self.input * R / ((self.input.shape[2]*self.input.shape[3]) * self.output + 1e-8)
        return lrp_input

    def _epsilon_lrp(self, R, epsilon, device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R,device=device)

    def _ww_lrp(self, R,device):
        '''
        There are no weights to use. default to _flat_lrp(R)
        '''
        return self._flat_lrp(R,device)

    def _flat_lrp(self, R, device):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''
        return self._simple_lrp(R,device=device)

    def _alphabeta_lrp(self, R, labels, device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R, labels, device)

    def _composite_lrp(self, R, labels, device, beta=0):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        
        
#         n,f,w,h = self.input.shape
        
#         act, indices = F.max_pool2d(self.input, w,w, 0, 1, False, True)
        
#         R = F.max_unpool2d(R.reshape(act.shape), indices, w, w, 1, self.input.shape)
#         return R
    
    
        
        x_p = torch.where(self.input<0, torch.zeros(1).to(device), self.input)
        x_n = self.input - x_p

        R = R.unsqueeze(-1).unsqueeze(-1)
        Rx_p = x_p * R / (x_p.sum(dim=(2,3),keepdim=True) + 1e-8)
        Rx_n = x_n * R / (x_n.sum(dim=(2,3),keepdim=True) + 1e-8)
        return (1-beta) * Rx_p + (beta * Rx_n)

    def _composite_new_lrp(self, R, labels, device):
        return self._composite_lrp(R, labels, device)