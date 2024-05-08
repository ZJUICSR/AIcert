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


import copy
from torch.autograd import Variable
from PIL import Image

import os
from os.path import join, dirname
from tqdm import tqdm
from .models import *
from .utils import softCrossEntropy
from .utils import one_hot_tensor
from .attack_methods import Attack_FeaScatter

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


def train_fun(epoch, net, args_dict, optimizer, trainloader, device):
    # print('\nEpoch: %d' % epoch)
    f_path = os.path.join(args_dict['model_dir'], 'latest')
    # print(f'f_path={f_path}*************')
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

    iterator = tqdm(trainloader, ncols=100, leave=False, desc=f'epoch_{epoch}')
    train_loss = list()
    train_acc = list()
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

        train_loss.append(float(loss))
        acc = get_acc(outputs, targets)
        train_acc.append(float(acc))
        iterator.set_postfix({"Loss": f'{np.array(train_loss).mean():.6f}',
                              "Acc": f'{np.array(train_acc).mean():.6f}'})

    cache_path = join('./output/cache/FeaSca/',str(args_dict['dataset'])+'_'+str(args_dict['modelname']))
    if epoch % args_dict['save_epochs'] == 0 or epoch >= args_dict['max_epoch'] - 2:
        # print('Saving..')
        f_path = os.path.join(args_dict['model_dir'], ('checkpoint-%s' % epoch))
        f_cache = os.path.join('./output/cache/FeaSca/',str(args_dict['dataset'])+'_'+str(args_dict['modelname']), ('checkpoint-%s' % epoch))
        state = {
            # 'net': net.state_dict(),
            'net': net.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args_dict['model_dir']):
            os.mkdir(args_dict['model_dir'])
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        torch.save(state, f_path)
        torch.save(state, f_cache)

    if epoch >= 0:
        print('Saving latest @ epoch %s..' % (epoch))
        f_path = os.path.join(args_dict['model_dir'], 'latest')
        f_cache = os.path.join('./output/cache/FeaSca/',str(args_dict['dataset'])+'_'+str(args_dict['modelname']), ('checkpoint-%s' % epoch))
        state = {
            # 'net': net.state_dict(),
            'net': net.state_dict(),
            'epoch': epoch,
            # 'optimizer': optimizer.state_dict()
            'optimizer': optimizer
        }
        if not os.path.isdir(args_dict['model_dir']):
            os.mkdir(args_dict['model_dir'])
        # if not os.path.isdir(f_cache):
        #     os.mkdir(f_cache)
        torch.save(state, f_path)
        torch.save(state, f_cache)

    # #######################################二者都是局部变量
    return round(np.array(train_acc).mean(), 6)


def RobustEnhance(model, args_dict, logging=None):
    '''
    :param model1: 模型
    :param args_dict: 该算法运行的参数
    :return: 加固后的模型model2，准确率acc
    '''
    torch.set_printoptions(threshold=10000)
    np.set_printoptions(threshold=np.inf)

    if 1:
        if args_dict['dataset'] == 'cifar10':
            logging.info('------------cifar10---------')
            args_dict['num_classes'] = 10
            args_dict['image_size'] = 32
        elif args_dict['dataset'] == 'cifar100':
            logging.info('----------cifar100---------')
            args_dict['num_classes'] = 100
            args_dict['image_size'] = 32
        if args_dict['dataset'] == 'svhn':
            logging.info('------------svhn10---------')
            args_dict['num_classes'] = 10
            args_dict['image_size'] = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # start_epoch = 0

    # Data
    logging.info('==> Preparing data..')

    transform_train = 0 # 定义全局变量，用来局部赋值
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
        trainset = torchvision.datasets.CIFAR10(root="dataset/CIFAR10",
                                                train=True,
                                                download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root="dataset/CIFAR10",
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck')
    elif args_dict['dataset'] == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root="dataset/CIFAR100",
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root="dataset/CIFAR100",
                                                train=False,
                                                download=True,
                                                transform=transform_test)
    elif args_dict['dataset'] == 'svhn':
        trainset = torchvision.datasets.SVHN(root="dataset/SVHN",
                                             split='train',
                                             download=True,
                                             transform=transform_train)
        testset = torchvision.datasets.SVHN(root="dataset/SVHN",
                                            split='test',
                                            download=True,
                                            transform=transform_test)

    # 加载训练数据
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args_dict['batch_size_train'],
                                              shuffle=True,
                                              num_workers=2)

    logging.info('==> Building model..')
    if args_dict['dataset'] == 'cifar10' or args_dict['dataset'] == 'cifar100' or args_dict['dataset'] == 'svhn':
        logging.info('---Reading model-----')
        basic_net = model

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

    if args_dict['adv_mode'].lower() == 'featurescatter':
        logging.info('-----Feature Scatter mode -----')
        net = Attack_FeaScatter(basic_net, config_feature_scatter)
    else:
        logging.info('-----OTHER_ALGO mode -----')
        raise NotImplementedError("Please implement this algorithm first!")

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(),
                          lr=args_dict['lr'],
                          momentum=args_dict['momentum'],
                          weight_decay=args_dict['weight_decay'])

    if args_dict['resume'] and args_dict['init_model_pass'] != '-1':
        # Load checkpoint.
        logging.info('==> Resuming from checkpoint..')
        f_cache_path = os.path.join('./output/cache/FeaSca/',str(args_dict['dataset'])+'_'+str(args_dict['modelname']), 'latest')
        f_path_latest = os.path.join(args_dict['model_dir'], 'latest')
        f_path = os.path.join(args_dict['model_dir'],
                              ('checkpoint-%s' % args_dict['init_model_pass']))
        # if not os.path.isdir(args_dict['model_dir']):
        #     logging.info('train from scratch: no checkpoint directory or file found')
        # elif args_dict['init_model_pass'] == 'latest' and os.path.isfile(f_path_latest):
        #     checkpoint = torch.load(f_path_latest)
        if args_dict['init_model_pass'] == 'latest' and os.path.isfile(f_cache_path):
            checkpoint = torch.load(f_cache_path)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info('resuming from epoch %s in latest' % start_epoch)
        else:
            start_epoch = 0
            logging.info('train from scratch: no checkpoint directory or file found')
        # elif os.path.isfile(f_cache_path):
        #     start_epoch = 0
        #     logging.info('train from scratch: no checkpoint directory or file found')
        # elif os.path.isfile(f_path):
        #     checkpoint = torch.load(f_path)
        #     net.load_state_dict(checkpoint['net'])
        #     start_epoch = checkpoint['epoch'] + 1
        #     logging.info('resuming from epoch %s' % (start_epoch - 1))
        # elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        #     logging.info('train from scratch: no checkpoint directory or file found')

    soft_xent_loss = softCrossEntropy()

    model2 = 0
    acc = 0
    logging.info(f'start_epoch={start_epoch}')
    loop = tqdm(range(start_epoch, args_dict['max_epoch']), ncols=100)
    for epoch in loop:
        acc = train_fun(epoch, net, args_dict, optimizer, trainloader, device)
        loop.set_postfix({"acc": f'{acc:.6f}'})

    return acc


# if __name__ == "__main__":

#     model = WideResNet((28, 10, 0.3, 10))
#     RobustEnhance(model1, args_dict)
