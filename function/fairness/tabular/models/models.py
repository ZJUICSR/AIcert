import imp
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

class LR(nn.Module) :
    def __init__(self, num_features) :
        super(LR, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_features, 2),
                                # nn.ReLU(),
                                # nn.Linear(4, 2)
                                )
    def forward(self, x) :
        x = torch.softmax(self.net(x), dim=1)
        return x #F.softmax(out)

class GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Net(nn.Module):

    def __init__(self, input_shape, output_shape=1, grl_lambda=100):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self._grl_lambda = grl_lambda
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, output_shape)
        if self._grl_lambda != 0:
            self.grl = GradientReversal(grl_lambda)
            self.fc5 = nn.Linear(32, 1) #! the second dimension of this layer is changed from 2 to 1
        # self.grl = GradientReversal(100)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, 0.1, training=self.training)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)

        y = self.fc4(hidden)
        # y = torch.softmax(y,dim=1)
        # y = F.dropout(y, 0.1)

        if self._grl_lambda != 0:
            s = self.grl(hidden)
            s = self.fc5(s)
            # s = F.sigmoid(s)
            # s = F.dropout(s, 0.1)
            return y, s
        else:
            return y