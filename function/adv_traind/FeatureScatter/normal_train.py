
import torch
from tqdm import tqdm
import os,shutil
from os.path import join, dirname
import numpy as np
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from .attack_methods import Attack_None, Attack_PGD
from .utils import CWLoss
import torch.backends.cudnn as cudnn
from copy import deepcopy


def train_single_epoch(model,
                       dataloader,
                       lr_scheduler,
                       loss_func,
                       optimizer,
                       epoch: int,
                       device='cuda'):
    model.train()
    criterion = loss_func
    train_loss = list()
    train_acc = list()
    loop = tqdm(enumerate(dataloader), ncols=100, desc=f'Train epoch {epoch}', total=len(dataloader),
                colour='blue', leave=False)
    for batch_idx, data in loop:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        mini_out = model(images)[0]
        mini_loss = criterion(mini_out, labels.long())
        mini_loss.backward()
        optimizer.step()

        _, pred = torch.max(mini_out, 1)
        acc = (pred.data == labels).float().mean()
        train_loss.append(float(mini_loss))
        train_acc.append(float(acc))

        loop.set_postfix({"Loss": f'{np.array(train_loss).mean():.6f}',
                          "Acc": f'{np.array(train_acc).mean():.6f}'})

    torch.cuda.empty_cache()
    lr_scheduler.step(epoch=epoch)
    return np.array(train_acc).mean(), np.array(train_loss).mean()


def train_model(model, dataloader, train_epoch, model_save_dir, device='cuda', model_name='x.pt'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)
    best_acc = 0
    for epoch in tqdm(range(train_epoch), ncols=100, desc=f'train'):
        acc, _ = train_single_epoch(model=model,
                                    dataloader=dataloader,
                                    lr_scheduler=lr_scheduler,
                                    loss_func=criterion,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    device=device)
        if acc > best_acc:
            # print('==> Saving checkpoints...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': 5,
            }
        torch.save(state, join(model_save_dir, model_name))

        if acc > 0.99:
            break
    return acc


def evaluate_robutness(net, testloader, device, args_dict):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    iterator = tqdm(testloader, ncols=200, leave=True)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)

        pert_inputs = inputs.detach()

        outputs, _ = net(pert_inputs, targets, batch_idx=batch_idx)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        iterator.set_description(
            str(predicted.eq(targets).sum().item() / targets.size(0)))

        if batch_idx % args_dict['log_step'] == 0:
            iterator.set_postfix({"acc": f'{correct_num / batch_size:.6f}',
                                  "avg-acc": f'{correct / total:.6f}'})
    acc = correct / total
    print('Val acc:', acc)
    return acc



def train_normal_model(model, args_dict, logging=None):
    '''
       :param model1: 模型
       :param args_dict: 该算法运行的参数
       :return: 加固后的模型model2，准确率acc
       '''
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(threshold=10000)
    np.set_printoptions(threshold=np.inf)

    model = deepcopy(model)

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

    # Data
    logging.info('==> Preparing data..')

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
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
        ])

    classes = 0
    if args_dict['dataset'] == 'cifar10':
        trainset = CIFAR10(root="dataset/CIFAR10", train=True, download=True, transform=transform_train)
        testset = CIFAR10(root="dataset/CIFAR10", train=False, download=True, transform=transform_test)
    elif args_dict['dataset'] == 'cifar100':
        trainset = CIFAR100(root="dataset/CIFAR100", train=True, download=True, transform=transform_train)
        testset = CIFAR100(root="dataset/CIFAR100", train=False, download=True, transform=transform_test)
    elif args_dict['dataset'] == 'svhn':
        trainset = SVHN(root="dataset/SVHN", split='train', download=True, transform=transform_train)
        testset = SVHN(root="dataset/SVHN", split='test', download=True, transform=transform_test)

    # 加载训练数据
    trainloader = DataLoader(trainset, batch_size=args_dict['batch_size_train'], shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args_dict['batch_size_train'], shuffle=True, num_workers=2)

    model_save_dir = args_dict['model_dir']
    epochs = args_dict['max_epoch']
    
    # 判断是否存在缓存模型，存在直接加载，不存在重新训练后保存
    cache_path = join('./output/cache/FeaSca/',str(args_dict['dataset'])+'_'+str(args_dict['modelname']))
    if os.path.exists(join(cache_path,'normal_train_model.pt')):
        check_state = torch.load(join(cache_path,'normal_train_model.pt'), map_location=device)
        model.load_state_dict(check_state['model'])
    else:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        train_model(model=model, dataloader=trainloader, train_epoch=epochs,
                model_save_dir= cache_path, device=device, model_name=f'normal_train_model.pt')
        shutil.copytree(cache_path, args_dict['model_dir'])
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

    # attack_list = args_dict['attack_method_list'].split('-')
    attack_list = args_dict['attack_method_list']
    attack_num = len(attack_list)
    attack_results = dict()
    for attack_idx in range(attack_num):
        args_dict['attack_method'] = attack_list[attack_idx]
        if args_dict['attack_method'] == 'Natural':
            logging.info('-----natural non-adv mode -----')
            net = Attack_None(model, config_natural)
        elif args_dict['attack_method'].upper() == 'FGSM':
            logging.info('-----FGSM adv mode -----')
            net = Attack_PGD(model, config_fgsm)
        elif args_dict['attack_method'].upper() == 'PGD':
            logging.info('-----PGD adv mode -----')
            net = Attack_PGD(model, config_pgd)
        elif args_dict['attack_method'].upper() == 'CW':
            logging.info('-----CW adv mode -----')
            net = Attack_PGD(model, config_cw)
        else:
            raise Exception(
                'Should be a valid attack method. The specified attack method is: {}'
                .format(args_dict['attack_method']))

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        acc = evaluate_robutness(net.to(device), testloader, device, args_dict)
        attack_results[args_dict['attack_method']] = acc
    logging.info(f'normal train result = {attack_results}')
    return attack_results


if __name__ == "__main__":
    pass

