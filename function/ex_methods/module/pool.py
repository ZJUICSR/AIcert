import torch

from .module import Module
#from torch.utils import _single, _pair, _triple
from torch.nn import functional as F


class _MaxPoolNd(Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_j, h, w)  = \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}
                               \text{input}(N_i, C_j, \text{stride}[0] * h + m, \text{stride}[1] * w + n)
        \end{equation*}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful when Unpooling later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding}[0] - \text{dilation}[0]
                    * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding}[1] - \text{dilation}[1]
                    * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def forward(self, input,interpreter='lrp'):
        self.input_tensor = input
        self.in_N, self.in_depth, self.in_h, self.in_w = self.input_tensor.shape
        
        if interpreter == 'lrp':
            self.activations, self.indices = F.max_pool2d(input, self.kernel_size, self.stride,
                                            self.padding, self.dilation, self.ceil_mode,
                                            True)
        else:
            self.activations, self.indices = F.max_pool2d(input, self.kernel_size, self.stride,
                                                        self.padding, self.dilation, self.ceil_mode,
                                                        True)
        return self.activations

    def clean(self):
        self.activations = None
        self.R = None
        
    def get_grad_and_activation(self, grad_output, requires_activation):
        '''
        dx: derivative of previous layer
        requires_activation: True if current layer is target layer.
        '''
        grad_input = F.max_unpool2d(grad_output.reshape(self.activations.shape), self.indices, self.kernel_size, self.stride,self.padding, self.input_tensor.shape)
        if requires_activation:
            return grad_input, self.activations
        else:
            return grad_input, None

    def _simple_lrp(self, R, labels, device):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        
        lrp_input = F.max_unpool2d(R.reshape(self.activations.shape), self.indices, self.kernel_size, self.stride, self.padding, self.input_tensor.shape)

        return lrp_input

    def _epsilon_lrp(self, R, epsilon, device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R,device)


    def _alphabeta_lrp(self, R, labels, device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R, labels,device)

    def _composite_lrp(self, R, labels,device):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R, labels,device)
   
    
    def check_shape(self, R):
        self.R = R
        R_shape = self.R.shape
        if len(R_shape) != 4:
            self.R = torch.reshape(self.R, self.activations.shape)
        N, NF, self.Hout, self.Wout = self.R.shape

    def extract_patches(self):
        '''
        <im2col>
        Apply sliding window to input volumes of convolution and make it as a matrix (each column represents receptive field)
        length of rows are the same as the stretched size of output feature map size. i.e. Hout x Wout
        (N, C_in, H_in, W_in) with filter size (Kernel_H, Kernel_W) -> (N, C_in x Kernel_H x Kernel_W, L)
        where L = Hout x Wout
        '''
        # image_patches = tf.extract_image_patches(self.input_tensor, ksizes=[1, self.pool_size, self.pool_size, 1],
        #                                          strides=[1, self.stride_size, self.stride_size, 1], rates=[1, 1, 1, 1],
        #                                          padding=self.pad)
        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride,
                                 dilation=self.dilation)
        image_patches = unfold(self.input_tensor)
        return image_patches


class _AvgPoolNd(Module):

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )


class AvgPool2d(_AvgPoolNd):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               \text{input}(N_i, C_j, \text{stride}[0] * h + m, \text{stride}[1] * w + n)
        \end{equation*}

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
        - Output: :math:`(N, C, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] -
                \text{kernel_size}[0]}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] -
                \text{kernel_size}[1]}{\text{stride}[1]} + 1\right\rfloor

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

    def forward(self, input):
        self.activations = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        return self.activations