import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

CONFIGS = {"num_classes": 1, "num_groups": 2, "num_epochs": 2000,
           "batch_size": 128, "lr": 1.0, "input_dim": 10,
           "hidden_layers": [32, 32, 32, 32], "adversary_layers": [32, 32], "grl_lambda": 0, "adversary_cls": 2}

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

    def __init__(self, input_shape, output_shape=None, grl_lambda=None, configs=CONFIGS):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.configs = configs
        self._grl_lambda = configs['grl_lambda'] if grl_lambda is None else grl_lambda
        
        self.input_dim = configs["input_dim"] if input_shape is None else input_shape
        self.num_classes = configs["num_classes"] if output_shape is None else output_shape
        self.adv_cls = configs["adversary_cls"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], self.num_classes)
        # Parameter of the adversary classification layer.

        
        # self.fc1 = nn.Linear(input_shape, 32)
        # self.fc2 = nn.Linear(32, 32)
        # self.fc3 = nn.Linear(32, 32)
        # self.fc4 = nn.Linear(32, output_shape)
        if self._grl_lambda != 0:
            self.grl = GradientReversal(grl_lambda)
            # self.fc5 = nn.Linear(32, 1) #! the second dimension of this layer is changed from 2 to 1
            self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
            self.num_adversaries_layers = len(configs["adversary_layers"])
            self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                            for i in range(self.num_adversaries_layers)])
            self.sensitive_cls = nn.Linear(self.num_adversaries[-1], self.adv_cls) #! the second dimension of this layer is changed from 2 to 1
        # self.grl = GradientReversal(100)

    def forward(self, x):
        # hidden = self.fc1(x)
        # hidden = F.relu(hidden)
        # hidden = F.dropout(hidden, 0.1, training=self.training)
        # hidden = self.fc2(hidden)
        # hidden = F.relu(hidden)
        # hidden = self.fc3(hidden)
        # hidden = F.relu(hidden)

        # y = self.fc4(hidden)
        # y = torch.softmax(y,dim=1)
        # y = F.dropout(y, 0.1)
        
        h_relu = x
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probability.
        # logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        logit = self.softmax(h_relu)
        logprobs = F.softmax(logit, dim=1) if self.num_classes > 1 else torch.sigmoid(logit)
        if self._grl_lambda != 0:
        # Adversary classification component.
            h_relu = self.grl(h_relu)
            for adversary in self.adversaries:
                h_relu = F.relu(adversary(h_relu))
            cls = F.log_softmax(self.sensitive_cls(h_relu), dim=1)
            return logprobs, cls
        else:
            return logprobs

        # if self._grl_lambda != 0:
        #     s = self.grl(hidden)
        #     s = self.fc5(s)
        #     # s = F.sigmoid(s)
        #     # s = F.dropout(s, 0.1)
        #     return y, s
        # else:
        #     return y
        
    @property
    def num_parameters(self):
        return sum([layer.weight.nelement() for layer in self.hiddens])
        # return self.fc1.weight.nelement() + self.fc2.weight.nelement()

    @property
    def weight(self):
        return torch.cat([layer.weight.flatten() for layer in self.hiddens], dim=0)
        # return torch.cat([self.fc1.weight.flatten(), self.fc2.weight.flatten()], dim=0)
        
class CFairNet(Net):
    def __init__(self, input_shape, output_shape=None, grl_lambda=None, configs=CONFIGS):
        super().__init__(input_shape, output_shape, grl_lambda, configs)
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1]) for i in range(self.num_adversaries_layers)]) for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])
        
    def forward(self, inputs, labels):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        if self._grl_lambda != 0:
            c_losses = []
            h_relu = self.grl(h_relu)
            for j in range(self.num_classes):
                idx = labels == j
                c_h_relu = h_relu[idx]
                for hidden in self.adversaries[j]:
                    c_h_relu = F.relu(hidden(c_h_relu))
                c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
                c_losses.append(c_cls)
            return logprobs, c_losses
        else:
            return logprobs

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs
        
class Net2(Net):
    def __init__(self, input_shape, output_shape=1, grl_lambda=100):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self._grl_lambda = grl_lambda
        self.fc1 = nn.Linear(input_shape, 32, bias=False)
        self.fc2 = nn.Linear(32, 32, bias=False)
        self.fc3 = nn.Linear(32, 32, bias=False)
        self.fc4 = nn.Linear(32, output_shape, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, 0.1, training=self.training)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)

        y = self.fc4(hidden)
        return self.sigmoid(y)
