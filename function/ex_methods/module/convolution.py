# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from .module import Module
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.activations_shape = None
        
        
        ### 181224-float16
        if transposed:
            tensor = torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            self.weight = Parameter(tensor.clone().detach())
        else:
            tensor = torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            self.weight = Parameter(tensor.clone().detach())
        if bias:
            tensor = torch.Tensor(out_channels)
            self.bias = Parameter(tensor.clone().detach())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_running_stats(self):
        
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
            
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def get_grad_and_activation(self, grad_output, requires_activation):
        '''
        dx: derivative of previous layer
        requires_activation: True if current layer is target layer.
        '''
        
        grad_input = F.grad.conv2d_input(self.input_tensor.shape, self.weight, grad_output, self.stride, self.padding, self.dilation, self.groups)
        if requires_activation:
            return grad_input, self.activations
        else:
            return grad_input, None
        
    
    def _simple_lrp(self, R, labels, device):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
#         print('cnn : simple lrp')
        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
        
        # Variables
        
        def f(layer,X):
            # Get activations of full positive or negative part.
            
            Zs = F.conv2d(X, layer, bias, stride,
                        padding, dilation, self.groups)

            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
            RdivZs = R/(Zs+stabilizer)


            tmp = torch.nn.grad.conv2d_input(X.shape, layer, grad_output=RdivZs, padding=1) * X

            
            return tmp
        r = f(weight,input_patches)
        return r
    

    
    def _composite_lrp(self, R, labels, device, beta=0):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        R = self.check_shape(R)

        input_patches = self.input_tensor
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
        alpha = 1 - beta
        beta = beta
        
        # Variables
        weight_p = F.relu(weight)
        weight_n = weight - weight_p
        
        
        input_p = F.relu(input_patches)
        input_n = input_patches - input_p
        
        def f(layer1, layer2, X1, X2):
            Z1 = F.conv2d(X1, layer1, bias, stride, padding, dilation, self.groups)
            Z2 = F.conv2d(X2, layer2, bias, stride, padding, dilation, self.groups)
            Zs = Z1 + Z2
            
            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            RdivZs = R/(Zs+stabilizer)

            tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
            tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

            return tmp1 + tmp2
        
        re_alpha = f(weight_p, weight_n, input_p, input_n)
        
        if  beta != 0:
            re_beta = f(weight_n, weight_p, input_p, input_n)
            re = (re_alpha * alpha) + (re_beta * beta)
        else:
            re = re_alpha
        
        return re
    


    def check_shape(self, R):
        self.R = R
        R_shape = self.R.shape
        if len(R_shape) != 4:
            self.R = torch.reshape(self.R, self.activations_shape)
        N, NF, self.Hout, self.Wout = self.R.shape
        return self.R
    
    
    def extract_patches(self):

        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        image_patches = unfold(self.input_tensor)

        '''
        <im2col>
        Apply sliding window to input volumes of convolution and make it as a matrix (each column represents receptive field)
        length of rows are the same as the stretched size of output feature map size. i.e. Hout x Wout
        (N, C_in, H_in, W_in) with filter size (Kernel_H, Kernel_W) -> (N, C_in x Kernel_H x Kernel_W, L)
        where L = Hout x Wout 
        '''

        return image_patches
    
    
    
class Conv1d(_ConvNd):
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor \frac{\text{out_channels}}{\text{in_channels}} \right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)


    def check_input_shape(self):
        inp_shape = self.input_tensor.shape
        try:
            if len(inp_shape)!=4:
                mod_shape = [self.batch_size, self.input_depth, self.input_dim, self.input_dim]
                self.input_tensor = torch.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError('Expected dimension of input tensor: 4')


    def forward(self, input):
        self.input_tensor = input
        self.check_input_shape()
        self.in_N, _, self.in_h, self.in_w = self.input_tensor.shape
        activations = F.conv2d(input=input,weight= self.weight, bias=self.bias,stride=self.stride,
                        padding= self.padding,dilation=self.dilation,groups=self.groups)
        self.activations = activations
        self.activations_shape = activations.shape
        
        return activations


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 3D `cross-correlation`_ operator

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class _ConvTransposeMixin(object):

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class ConvTranspose1d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding}
                    + \text{kernel_size} + \text{output_padding}

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)



class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0]
                    + \text{kernel_size}[0] + \text{output_padding}[0]

              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1]
                    + \text{kernel_size}[1] + \text{output_padding}[1]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)



class ConvTranspose3d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 3D transposed convolution operator over an input image composed of several input
    planes.
    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.

    This module can be seen as the gradient of Conv3d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0]
                    + \text{kernel_size}[0] + \text{output_padding}[0]

              H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1]
                    + \text{kernel_size}[1] + \text{output_padding}[1]

              W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2]
                    + \text{kernel_size}[2] + \text{output_padding}[2]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)