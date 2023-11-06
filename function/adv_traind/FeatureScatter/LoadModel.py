from __future__ import print_function
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.autograd.gradcheck import zero_gradients
import copy
from torch.autograd import Variable
from PIL import Image

import os
import argparse
import datetime

from tqdm import tqdm

from models.lenet import LeNet
from models import *

import utils
from utils import softCrossEntropy
from utils import one_hot_tensor
from attack_methods import Attack_FeaScatter


def print_para(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
        break


def get_acc(outputs, targets):
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    acc = 1.0 * correct / total
    return acc


def train_fun(epoch, net, args, optimizer, trainloader, device):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if epoch < args_dict['decay_epoch1']:
        lr = args_dict['lr']
    elif epoch < args_dict['decay_epoch2']:
        lr = args_dict['lr'] * args_dict['decay_rate']
    else:
        lr = args_dict['lr'] * args_dict['decay_rate'] * args_dict['decay_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc

    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        adv_acc = 0

        optimizer.zero_grad()

        # forward
        outputs, loss_fs = net(inputs.detach(), targets)

        optimizer.zero_grad()
        loss = loss_fs.mean()
        loss.backward()

        optimizer.step()

        train_loss = loss.item()

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:
            if adv_acc == 0:
                adv_acc = get_acc(outputs, targets)
            iterator.set_description(str(adv_acc))

            nat_outputs, _ = net(inputs, targets, attack=False)
            nat_acc = get_acc(nat_outputs, targets)

            print(
                "epoch %d, step %d, lr %.4f, duration %.2f, training nat acc %.2f, training adv acc %.2f, training adv loss %.4f"
                % (epoch, batch_idx, lr, duration, 100 * nat_acc,
                   100 * adv_acc, train_loss))

    if epoch % args.save_epochs == 0 or epoch >= args_dict['max_epoch'] - 2:
        print('Saving..')
        f_path = os.path.join(args_dict['model_dir'], ('checkpoint-%s' % epoch))
        state = {
            # 'net': net.state_dict(),
            'net': net,
            # 'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args_dict['model_dir']):
            os.mkdir(args_dict['model_dir'])
        torch.save(state, f_path)

    if epoch >= 0:
        print('Saving latest @ epoch %s..' % (epoch))
        f_path = os.path.join(args_dict['model_dir'], 'latest')
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'epoch': epoch,
            # 'optimizer': optimizer.state_dict()
            'optimizer': optimizer
        }
        if not os.path.isdir(args_dict['model_dir']):
            os.mkdir(args_dict['model_dir'])
        torch.save(state, f_path)

    # #######################################二者都是局部变量
    return state, nat_acc


def RobustEnhance(args_dict):

    parser = argparse.ArgumentParser(description='Feature Scatterring Training')
    parser.register('type', 'bool', utils.str2bool)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--adv_mode', default='feature_scatter', type=str, help='adv_mode (feature_scatter)')
    parser.add_argument('--save_epochs', default=100, type=int, help='save period')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (1-tf.momentum)')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--log_step', default=10, type=int, help='log_step')
    args = parser.parse_args()

    torch.set_printoptions(threshold=10000)
    np.set_printoptions(threshold=np.inf)

    if 1:
        if args_dict['dataset'] == 'cifar10':
            print('------------cifar10---------')
            args_dict['num_classes'] = 10
            args_dict['image_size'] = 32
        elif args_dict['dataset'] == 'cifar100':
            print('----------cifar100---------')
            args_dict['num_classes'] = 100
            args_dict['image_size'] = 32
        if args_dict['dataset'] == 'svhn':
            print('------------svhn10---------')
            args_dict['num_classes'] = 10
            args_dict['image_size'] = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0

    # Data
    print('==> Preparing data..')

    transform_train = 0  # 定义全局变量，用来局部赋值
    transform_test = 0
    if args_dict['dataset'] == 'cifar10' or args_dict['dataset'] == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])
    elif args_dict['dataset'] == 'svhn':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])

    trainset = 0
    testset = 0
    classes = 0
    if args_dict['dataset'] == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/data/user/WZT/Datasets/cifar10/',
                                                train=True,
                                                download=False,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/data/user/WZT/Datasets/cifar10/',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck')
    elif args_dict['dataset'] == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/data/user/WZT/Datasets/cifar100/',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/data/user/WZT/Datasets/cifar100/',
                                                train=False,
                                                download=True,
                                                transform=transform_test)
    elif args_dict['dataset'] == 'svhn':
        trainset = torchvision.datasets.SVHN(root='/data/user/WZT/Datasets/svhn/',
                                             split='train',
                                             download=True,
                                             transform=transform_train)
        testset = torchvision.datasets.SVHN(root='/data/user/WZT/Datasets/svhn/',
                                            split='test',
                                            download=True,
                                            transform=transform_test)

    # 加载训练数据
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args_dict['batch_size_train'],
                                              shuffle=True,
                                              num_workers=2)

    print('==> Building model..')
    if args_dict['dataset'] == 'cifar10' or args_dict['dataset'] == 'cifar100' or args_dict['dataset'] == 'svhn':
        print('---Reading model-----', args_dict['model1'])
        basic_net = args_dict['model1']

    basic_net = basic_net.to(device)

    # config for feature scatter
    config_feature_scatter = {
        'train': True,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'random_start': True,
        'ls_factor': 0.5,
    }

    if args.adv_mode.lower() == 'feature_scatter':
        print('-----Feature Scatter mode -----')
        net = Attack_FeaScatter(basic_net, config_feature_scatter)
    else:
        print('-----OTHER_ALGO mode -----')
        raise NotImplementedError("Please implement this algorithm first!")

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(),
                          lr=args_dict['lr'],
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args_dict['resume'] and args_dict['init_model_pass'] != '-1':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        f_path_latest = os.path.join(args_dict['model_dir'], 'latest')
        f_path = os.path.join(args_dict['model_dir'],
                              ('checkpoint-%s' % args_dict['init_model_pass']))
        if not os.path.isdir(args_dict['model_dir']):
            print('train from scratch: no checkpoint directory or file found')
        elif args_dict['init_model_pass'] == 'latest' and os.path.isfile(f_path_latest):
            checkpoint = torch.load(f_path_latest)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            print('resuming from epoch %s in latest' % start_epoch)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            print('resuming from epoch %s' % (start_epoch - 1))
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print('train from scratch: no checkpoint directory or file found')

    soft_xent_loss = softCrossEntropy()

    model2 = 0
    acc = 0
    for epoch in range(start_epoch, args_dict['max_epoch']):
        model2, acc = train_fun(epoch, net, args, optimizer, trainloader, device)

    return model2, acc


if __name__ == "__main__":
    args_dict = {
        'model_dir': '/data/user/WZT/models/feasca_cifar10_letnet/',  # 模型训练输出路径及加载路径
        'init_model_pass': 'checkpoint-0',  # 默认'-1'， 加,路径为model_dir。(-1: from scratch; K: checkpoint-K; latest = latest)
        'resume': True, # 加载init_model_pass 继续训练
        'lr': 0.1,  # 学习率
        'batch_size_train': 128,  # 每次train的样本数
        'max_epoch': 200, # 训练批次大小
        'decay_epoch1': 60,
        'decay_epoch2': 90,
        'decay_rate': 0.1,
        'dataset': 'cifar10',  # 数据集选择（cifar10，cifar100，svhn）
        'num_classes': 10,  # 图片种类（cifar10 = 10，cifar100 =100，svhn =10）
        'image_size': 32, # 数据集样本规格大小
        'model1': LeNet(10)  # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
    }

    RobustEnhance(args_dict)
