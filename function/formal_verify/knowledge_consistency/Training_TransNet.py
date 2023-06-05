import argparse
import os
import random
import shutil
import time
import warnings
import json
from xml.dom.pulldom import default_bufsize

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from logger import Logger
import Model_zoo as models
from Datasets import DiscreteDataset


np.set_printoptions(precision=3)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--alphas', default='[0.5, 0.5]', type=str)
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--dataset', default='CUB200', type=str)
parser.add_argument('--epoch_step', default=60, type=int)
parser.add_argument('--save_per_epoch',default=200, action='store_true')
parser.add_argument('--save_epoch', default=100, type=int)
parser.add_argument('--logspace', default=2, type=int)
parser.add_argument('--conv_layer', default=30, type=int)
parser.add_argument('--fine_tune', default=False, type=bool)
parser.add_argument('--decay_factor', default=0.2, type=float)
parser.add_argument('--device_ids', default='[2,5]', type=str)
parser.add_argument('--sample_num', default='', type=str)
parser.add_argument('--convOut_path',
                    default= 'convOuts/',
                    type=str)
# parser.add_argument('--val_path',
#                     default= 'convOut_VOC2012_crop_vgg16_bn_ft_L40_20par_val_v2.pkl',
#                     type=str)

parser.add_argument('--validate', action='store_true')
parser.add_argument('--model', default='sigmoid_p_Instance', type=str)
parser.add_argument('--layers', default=3, type=int)
parser.add_argument('--L2', action='store_false', help='if  add, then no L2 ')
parser.add_argument('--fix_p', action='store_true')
parser.add_argument('--bn', action='store_true')

parser.add_argument('--fc', action='store_true')

parser.add_argument('--affine', action='store_true')
parser.add_argument('--modified_norm', action='store_false')
best_acc1 = 0

opt = args = parser.parse_args()
device_ids = json.loads(args.device_ids)
alphas = torch.tensor(json.loads(args.alphas)).cuda(args.gpu)


print('parsed options:', vars(args))
print(alphas)

def main():
    global args, best_acc1
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.arch.startswith('vgg'):
        if args.conv_layer <= 30:
            input_size = output_size = torch.zeros((512, 28, 28)).shape
        else:
            input_size = output_size = torch.zeros((512, 14, 14)).shape
    elif args.arch.startswith('alexnet'):
        if args.conv_layer == 8 :
            input_size = output_size = torch.zeros((256, 13, 13)).shape
        elif args.conv_layer == 10 :
            input_size = output_size = torch.zeros((256, 13, 13)).shape
    elif args.arch.startswith('resnet'):
        if args.arch.startswith('resnet18') or args.arch.startswith('resnet34'):
            if args.conv_layer == 3 :
                input_size = output_size = torch.zeros((256, 14, 14)).shape
            elif args.conv_layer == 4 :
                input_size = output_size = torch.zeros((512, 7, 7)).shape
        else:
            if args.conv_layer == 3 :
                input_size = output_size = torch.zeros((1024, 14, 14)).shape
            elif args.conv_layer == 4 :
                input_size = output_size = torch.zeros((2048, 7, 7)).shape

    if args.fc:
        input_size = output_size = torch.zeros((4096, 1, 1)).shape

    model = models.LinearTester(input_size, output_size, gpu_id=args.gpu, fix_p=args.fix_p, bn=args.bn,
                         affine=args.affine, instance_bn=args.modified_norm)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda(device_ids[0])
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        args.gpu = device_ids[0]
        print('GPU used: ' + args.device_ids)
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            model.cuda(device_ids[0])
        else:
            model = torch.nn.DataParallel(model).cuda(device_ids[0])


    train_dataset = DiscreteDataset(dataset=args.dataset, layer=args.conv_layer, path=args.convOut_path)
    if args.sample_num != '':
        if os.path.exists("sub_sampler_{}_{}_A.npy".format(args.dataset, args.sample_num)):
            print("Find sub_sampler_{}_{}_A.npy!".format(args.dataset, args.sample_num))
            sub_idx = np.load("sub_sampler_{}_{}_A.npy".format(args.dataset, args.sample_num)).tolist()
        else:
            print("Can't find sub_sampler_{}_{}_A.npy!".format(args.dataset, args.sample_num))
            return
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(sub_idx))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    logger_train = Logger(
        './logs_M_multilayers/{}_{}/{}_L{}_{}/train'.format(args.dataset, args.arch, args.sample_num, args.conv_layer, args.suffix))
    logger_val = Logger(
        './logs_M_multilayers/{}_{}/{}_L{}_{}/val'.format(args.dataset, args.arch, args.sample_num, args.conv_layer, args.suffix))
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(args.gpu)))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        if args.logspace != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]
        else:
            adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger_train)

        is_best = False
        save_dir = './model_checkpoints/{}_{}/checkpoint_L{}_{}_{}.pth.tar'.format(args.dataset, args.arch, args.conv_layer, args.sample_num, args.suffix)
        os.makedirs('./model_checkpoints/{}_{}/'.format(args.dataset, args.arch), exist_ok=True)
        if epoch % 100 == 99:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                #             'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_dir)
        if args.save_per_epoch and epoch > 0 and epoch % args.save_epoch == 0:
            save_dir_itr = 'checkpoint_{}_{}_ep{}.pth.tar'.format(args.dataset, args.suffix, epoch)
            shutil.copyfile(save_dir, save_dir_itr)


def validate(train_loader, netA_part, netB_part, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mse = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():

        end = time.time()
        for i, (inp, tar) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                input = netA_part(inp).cuda(args.gpu, non_blocking=True)
                target = netB_part(inp).cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            if args.L2:
                mse.update(loss.item(), input.size(0))
                loss = loss + torch.sum((model.nonLinearLayers_p ** 2) * alphas)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    loss=losses, ))

        if args.L2:
            log_dict = {'Loss': losses.avg, 'MSE': mse.avg, 'L2_loss': losses.avg - mse.avg, }
        else:
            log_dict = {'Loss': losses.avg, 'MSE': losses.avg, 'L2_loss': 0.0, }
        set_tensorboard(log_dict, epoch, logger)
    return losses.avg


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ps = AverageMeter(size=(2,))
    dps = AverageMeter(size=(2,))
    mse = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, bs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        f_input = bs['convOut1'].cuda(args.gpu)
        target = bs['convOut2'].cuda(args.gpu)

        output = model(f_input)

        loss = criterion(output, target)
        if args.L2:
            mse.update(loss.item(), f_input.size(0))
            loss = loss + torch.sum((model.nonLinearLayers_p ** 2) * alphas)
        # measure accuracy and record loss
        losses.update(loss.item(), f_input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if not args.fix_p:
            if args.model.startswith('sigmoid_p'):
                dpp = model.nonLinearLayers_p_pre.grad.data.cpu().numpy()
                px = model.nonLinearLayers_p.data.cpu().numpy()
                dps.update(dpp / (px * (1 - px)), f_input.size(0))
            else:
                dps.update(model.nonLinearLayers_p.grad.data.cpu().numpy(), input.size(0))
        ps.update(model.nonLinearLayers_p.data.cpu().numpy())
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                loss=losses, ))

            print('\tP {}({})\t'
                  'DP {}({})'.format(ps.val, ps.avg,
                                     dps.val, dps.avg, ))

    if args.L2:
        log_dict = {'Loss': losses.avg, 'MSE': mse.avg, 'L2_loss': losses.avg - mse.avg,
                    'p1': ps.avg[0], 'p2': ps.avg[1],
                    'dps1': dps.avg[0], 'dps2': dps.avg[1],
                    'dps_a1': dps.avg[0] - 2 * ps.avg[0] * alphas[0],
                    'dps_a2': dps.avg[1] - 2 * ps.avg[1] * alphas[1],
                    }
    else:
        log_dict = {'Loss': losses.avg, 'MSE': losses.avg, 'L2_loss': 0.0,
                    'p1': ps.avg[0], 'p2': ps.avg[1],
                    'dps1': dps.avg[0], 'dps2': dps.avg[1]
                    }
    #set_tensorboard(log_dict, epoch, logger)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}_{}.pth.tar'.format(args.dataset, args.suffix))


# For tensorboard
def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)

    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, size=(1,)):
        self.size = size
        self.reset()

    def reset(self):
        self.val = 0

        if self.size != (1,):
            self.sum = np.zeros(self.size)
            self.avg = np.zeros(self.size)
        else:
            self.sum = 0
            self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.decay_factor ** (epoch // args.epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class fc_out(nn.Module):
    def __init__(self, fea, cla):
        super(fc_out, self).__init__()
        self.fea = fea
        self.cla = cla
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.fea(x)
            x = x.view(x.size(0), -1)
            x = self.cla(x)
            return x


def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(opt.gpu)))
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module'):
                state_dict[key.replace('module.', '')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
              .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    del checkpoint, state_dict


if __name__ == '__main__':
    main()
