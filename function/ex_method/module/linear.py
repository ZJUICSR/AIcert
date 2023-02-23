import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from .module import Module


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, lastLayer=False, bias=True, whichScore = None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.whichScore  = whichScore
        self.lastLayer = lastLayer
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
  
    def type_(self):
        return str('Linear')

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.kaiming_normal_(self.weight.data, nonlinearity='relu')
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def check_input_shape(self, input):
        inp_shape = input.shape
        if len(inp_shape)!=2:
            input = input.view(input.size(0),-1)
            
        return input

    def forward(self, input):

        self.input_tensor = self.check_input_shape(input)
        
        activation = F.linear(self.input_tensor, self.weight, self.bias)
        self.activation_shape = torch.Tensor(activation.shape)
        
        return activation

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def _grad_cam(self, grad_output, requires_activation):
        '''
        dx: derivative of previous layer
        requires_activation: True if current layer is target layer. In linear case, this variable is always false
        '''
        grad_input = grad_output.mm(self.weight)
        if requires_activation:
            return grad_input, self.input_tensor
        else:
            return grad_input, None
   
    def _simple_lrp(self, R, labels, device, epsilon=0.01):
        print('linear : simple lrp222')      
        # Variables
             
        Zs = F.linear(self.input_tensor, self.weight, self.bias)
        stabilizer = epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs += stabilizer

        
        if self.lastLayer and self.whichScore is None:
            self.whichScore = labels   
        if self.lastLayer and self.whichScore is not None:
            print('---last layer---')
            mask = torch.zeros_like(R)
            index = torch.range(0,R.shape[0]-1,dtype=torch.long).to(device)
            mask[index,self.whichScore] = 1
            R = R * mask


        RdivZs = R/Zs

        tmp1 = RdivZs.mm(self.weight)* self.input_tensor
        return tmp1 

    
    ## 190105 even though _alpha_beta_lrp() is called, _simple_lrp is runned ## 
    def _composite_lrp(self, R, labels, device, epsilon=0.01):
        Zs = F.linear(self.input_tensor, self.weight, self.bias) 
        stabilizer = epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs = Zs + stabilizer
        
        if self.lastLayer:
#             print('---last layer---')
            mask = torch.zeros_like(R)
            
            index = torch.arange(0, R.shape[0], dtype=torch.long).to(device)
            mask[index, labels] = 1
            R = R * mask

        RdivZs = R/Zs

        tmp1 = RdivZs.mm(self.weight)* self.input_tensor
        
        return tmp1