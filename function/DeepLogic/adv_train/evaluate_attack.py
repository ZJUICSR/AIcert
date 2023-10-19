from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from models.vggnet import *
from models.resnet import *
from models.densenet import *
from models.mynet import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--dataset', default='cifar10', help='use what dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model',
                    default='resnet18',
                    help='model name for evaluation')
parser.add_argument('--model-path',
                    default='./model-cifar-vggNet/model-wideres-epoch58.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
# set up data loader
if args.dataset == 'cifar10':
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
if args.dataset == 'fashionminist' and args.model!='densenet121':
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
if args.dataset == 'fashionminist' and args.model=='densenet121':
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=resize_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  random=True):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return err_pgd

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss
    
def _cw_whitebox(model,
                 X,
                 y,
                 epsilon=args.epsilon,
                 num_steps=args.num_steps,
                 step_size=args.step_size,
                 random=True):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = CWLoss(10)(model(X_pgd), y)   #cifar10--num_classes=10,imagnet--num_classes=1000
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err cw (white-box): ', err_pgd)
    return err_pgd

def eval_adv_test_whitebox_pgd20(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        if args.dataset=='fashionminist' and args.model!='densenet121':              
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            data = np.array(data)
            data = data.transpose((1, 0, 2, 3))  # array 转置
            data = np.concatenate((data, data, data), axis=0)  # 维度拼接
            data = data.transpose((1, 0, 2, 3))  # array 转置回来
            data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor         
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
    print('PGD20 robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_various_epsilon(model, device, test_loader):
    model.eval()
    test_epsilon = [2.0/255, 4.0/255, 8.0/255, 16.0/255, 32.0/255]
    for epsilon in test_epsilon:
        robust_err_total = 0
        for data, target in test_loader:
            if args.dataset=='fashionminist' and args.model!='densenet121':              
                # 因为FashionMNIST输入为单通道图片，需要转换为三通道
                data = np.array(data)
                data = data.transpose((1, 0, 2, 3))  # array 转置
                data = np.concatenate((data, data, data), axis=0)  # 维度拼接
                data = data.transpose((1, 0, 2, 3))  # array 转置回来
                data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor             
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_robust = _pgd_whitebox(model, X, y, epsilon=epsilon, step_size=epsilon/10.0)
            robust_err_total += err_robust
        print('PGD20 epsilon %.4f robust_acc: '% (epsilon), 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_pgd100(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        if args.dataset=='fashionminist' and args.model!='densenet121':              
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            data = np.array(data)
            data = data.transpose((1, 0, 2, 3))  # array 转置
            data = np.concatenate((data, data, data), axis=0)  # 维度拼接
            data = data.transpose((1, 0, 2, 3))  # array 转置回来
            data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor         
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, epsilon=args.epsilon,
                  num_steps=100, step_size=args.step_size)
        robust_err_total += err_robust
    print('PGD100 robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_fgsm(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0

    for data, target in test_loader:
        if args.dataset=='fashionminist' and args.model!='densenet121':              
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            data = np.array(data)
            data = data.transpose((1, 0, 2, 3))  # array 转置
            data = np.concatenate((data, data, data), axis=0)  # 维度拼接
            data = data.transpose((1, 0, 2, 3))  # array 转置回来
            data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor         
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, epsilon=8/255.0, num_steps=1,
                                step_size=8/255.0)
        robust_err_total += err_robust
    print('FGSM robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_cw(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        if args.dataset=='fashionminist' and args.model!='densenet121':              
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            data = np.array(data)
            data = data.transpose((1, 0, 2, 3))  # array 转置
            data = np.concatenate((data, data, data), axis=0)  # 维度拼接
            data = data.transpose((1, 0, 2, 3))  # array 转置回来
            data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor         
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _cw_whitebox(model, X, y)
        robust_err_total += err_robust
    print('cw robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))


def clean_test(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    err_total = 0
    for data, target in test_loader:
        if args.dataset=='fashionminist' and args.model!='densenet121':              
            # 因为FashionMNIST输入为单通道图片，需要转换为三通道
            data = np.array(data)
            data = data.transpose((1, 0, 2, 3))  # array 转置
            data = np.concatenate((data, data, data), axis=0)  # 维度拼接
            data = data.transpose((1, 0, 2, 3))  # array 转置回来
            data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor 
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out= model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        err_total += err
    print('Clean Acc: ', 1 - err_total / len(test_loader.dataset))
 

if __name__=='__main__':

    if args.white_box_attack:
        # white-box attack
        print('white-box attack')
        if args.model=='resnet18':
            model = ResNet18().to(device)
        elif args.model=='vgg16':
            model = vgg16_bn().to(device)
        elif args.model=='resnet34':
            model = ResNet34().to(device)
        elif args.model=='densenet121':
            model = DenseNet121(num_classes=10, grayscale=True).to(device)    
        elif args.model=='smallcnn':
            model = SmallCNN().to(device)
        model.load_state_dict(torch.load(args.model_path))
        clean_test(model, device, test_loader)
        # eval_adv_various_epsilon(model, device, test_loader)
        eval_adv_test_whitebox_pgd20(model, device, test_loader)
        # eval_adv_test_whitebox_pgd100(model, device, test_loader)
        eval_adv_test_whitebox_fgsm(model, device, test_loader)
        eval_adv_test_whitebox_cw(model, device, test_loader)
        
