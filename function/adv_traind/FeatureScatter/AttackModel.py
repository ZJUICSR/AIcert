from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from tqdm import tqdm
from .attack_methods import Attack_None, Attack_PGD
from .utils import softCrossEntropy, CWLoss, str2bool

global criterion


def test(epoch, net, args, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    iterator = tqdm(testloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        pert_inputs = inputs.detach()

        outputs, _ = net(pert_inputs, targets, batch_idx=batch_idx)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        duration = time.time() - start_time

        _, predicted = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        iterator.set_description(
            str(predicted.eq(targets).sum().item() / targets.size(0)))

        if batch_idx % args.log_step == 0:
            print(
                "step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                % (batch_idx, duration, 100. * correct_num / batch_size,
                   100. * correct / total, test_loss / total))

    acc = 100. * correct / total
    print('Val acc:', acc)
    return acc


def RobustTest(args_dict):
    parser = argparse.ArgumentParser(description='Feature Scattering Adversarial Training')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--resume', '-r', default=True, action='store_true', help='resume from checkpoint')
    parser.add_argument('--log_step', default=7, type=int, help='log_step')
    args = parser.parse_args()
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
    elif args_dict['dataset'] == 'mnist':
        print('----------mnist---------')
        args_dict['num_classes'] = 10
        args_dict['image_size'] = 28

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0
    # Data
    print('==> Preparing data..')
    transform_test = 0
    testset = 0
    # testloader = 0
    if args_dict['dataset'] == 'cifar10' or args_dict['dataset'] == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])
    elif args_dict['dataset'] == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])

    if args_dict['dataset'] == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='/home/Wenjie/AI-Platform/dataset/CIFAR10',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
    elif args_dict['dataset'] == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='/home/Wenjie/AI-Platform/dataset/CIFAR100',
                                                train=False,
                                                download=True,
                                                transform=transform_test)

    elif args_dict['dataset'] == 'svhn':
        testset = torchvision.datasets.SVHN(root='/home/Wenjie/AI-Platform/dataset/SVHN',
                                            split='test',
                                            download=True,
                                            transform=transform_test)
    # 加载测试数据
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args_dict['batch_size_test'],
                                             shuffle=False,
                                             num_workers=2)

    print('==> Building model..')
    if args_dict['dataset'] == 'cifar10' or args_dict['dataset'] == 'cifar100' or args_dict['dataset'] == 'svhn':
        # print('---Reading model-----', args_dict['model2'])
        # basic_net = WideResNet(depth=28,
        #                        num_classes=args.num_classes,
        #                        widen_factor=10)
        basic_net = args_dict['model2']
    basic_net = basic_net.to(device)
    # configs
    config_natural = {'train': False}

    config_fgsm = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 1,
        'step_size': 8.0 / 255 * 2,
        'random_start': True
    }

    config_pgd = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 20,
        'step_size': 2.0 / 255 * 2,
        'random_start': True,
        'loss_func': torch.nn.CrossEntropyLoss(reduction='none')
    }

    config_cw = {
        'train': False,
        'targeted': False,
        'epsilon': 8.0 / 255 * 2,
        'num_steps': 20,
        'step_size': 2.0 / 255 * 2,
        'random_start': True,
        'loss_func': CWLoss(args_dict['num_classes'])
    }

    attack_list = args_dict['attack_method_list'].split('-')
    attack_num = len(attack_list)
    acc_list = {}
    for attack_idx in range(attack_num):

        args_dict['attack_method'] = attack_list[attack_idx]

        if args_dict['attack_method'] == 'Natural':
            print('-----natural non-adv mode -----')
            # config is only dummy, not actually used
            net = Attack_None(basic_net, config_natural)
        elif args_dict['attack_method'].upper() == 'FGSM':
            print('-----FGSM adv mode -----')
            net = Attack_PGD(basic_net, config_fgsm)
        elif args_dict['attack_method'].upper() == 'PGD':
            print('-----PGD adv mode -----')
            net = Attack_PGD(basic_net, config_pgd)
        elif args_dict['attack_method'].upper() == 'CW':
            print('-----CW adv mode -----')
            net = Attack_PGD(basic_net, config_cw)
        else:
            raise Exception(
                'Should be a valid attack method. The specified attack method is: {}'
                .format(args_dict['attack_method']))

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume and args_dict['init_model_pass'] != '-1':
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            f_path_latest = os.path.join(args_dict['model_dir'], 'latest')
            # f_path = os.path.join(args_dict['model_dir'], ('checkpoint-%s' % args_dict['init_model_pass']))
            f_path = os.path.join(args_dict['model_dir'], args_dict['init_model_pass'])
            if not os.path.isdir(args_dict['model_dir']):
                print('train from scratch: no checkpoint directory or file found')
            elif args_dict['init_model_pass'] == 'latest' and os.path.isfile(f_path_latest):
                checkpoint = torch.load(f_path_latest)
                net.load_state_dict(checkpoint['net'])
                start_epoch = checkpoint['epoch']
                print('resuming from epoch %s in latest' % start_epoch)
            elif os.path.isfile(f_path):
                checkpoint = torch.load(f_path)
                net.load_state_dict(checkpoint['net'])
                start_epoch = checkpoint['epoch']
                print('resuming from epoch %s' % start_epoch)
            elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
                print('train from scratch: no checkpoint directory or file found')

        global criterion
        criterion = nn.CrossEntropyLoss()

        acc = test(0, net, args, testloader, device)
        acc_list[args_dict['attack_method']] = acc
    return acc_list


# if __name__ == "__main__":
#     args_dict = {
#         'attack': True,
#         'model2': LeNet(10),
#         # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
#         'model_dir': 'feasca_cifar10_lenet_result',  # 模型路径
#         'init_model_pass': 'latest',  # 加载文件名latest,路径为model_dir，
#         'attack_method': 'pgd',  # adv_mode : natural, PGD or CW
#         'attack_method_list': 'natural-PGD-CW-fgsm',# 无任何攻击下的acc和三种攻击下的acc
#         'dataset': 'cifar10',  # 数据集cifar10, cifar100,svhn
#         'image_size': 32,  # cifar10, cifar100,svhn, = 32; minist = 28
#         'num_classes': 10,  # 图片种类
#         'batch_size_test': 100  # 每次测试的样本数

#     }

#     RobustTest(args_dict)
