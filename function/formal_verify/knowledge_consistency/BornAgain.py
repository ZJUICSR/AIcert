import argparse
import os
import random
import shutil
import time
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from logger import Logger
from Datasets import Generate_Dataloader
from Models import Generate_Model
from torch.nn.modules.loss import _Loss

'''
KDLoss: 
    Args for forward: 
        input: tensor from different branches   shape: (num_teacher,num_sample, channel, w, h)
        target: tensor from different teacher(Over-fitting) Net
'''
class KDLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction=None):
        super(KDLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, tau=1):
        assert(len(input)==len(target))
        bs = input.shape[0]
        log_prob = nn.LogSoftmax()(input/tau)
        soft_tar = nn.Softmax()(target/tau)
        return -tau**2*torch.sum(log_prob*soft_tar)/bs


parser = argparse.ArgumentParser(description='Born Again Network')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--train_path', default='../CUB_200_2011/crop/train', help='../../ILSVRC2012/', type=str)
parser.add_argument('--val_path', default='../CUB_200_2011/crop/test', type=str, help='../ILSVRC2012_img_val')
parser.add_argument('--data_path', default='/home/data/lilongfei/VOCdevkit/VOC2012/OurFiles/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn')
parser.add_argument('--optim', default='SGD',type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--dataset', default='CUB200', type=str)
parser.add_argument('--epoch_step', default=60, type=int)
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--logspace', default=0, type=int)
parser.add_argument('--sample_num',default='',type=str)
parser.add_argument('--decay_factor', default=0.3, type=float)
parser.add_argument('--device_ids', default='[0,1,2,3]', type=str)
parser.add_argument('--train_layer', default=24, type=int)
parser.add_argument('--generations','-g', default=5, type=int)
parser.add_argument('--start_gen', default=0, type=int)
parser.add_argument('--lambd', default=0.5, type=float)
parser.add_argument('--lambd_end', default=0.5, type=float)
parser.add_argument('--tau', default=1.0, type=float)
parser.add_argument('--gpu_teacher', default=None, type=int)

args = parser.parse_args()
device_ids = json.loads(args.device_ids)
print('parsed options:', vars(args))
gpu = args.gpu = device_ids[0]
# Logspace lr
logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)
logspace_lambd = np.geomspace(args.lambd, args.lambd_end, args.epochs)

def main():
    global args, device_ids, gpu
    seed = args.seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Create dataloader
    train_loader, val_loader = \
        Generate_Dataloader(args.dataset, args.batch_size, args.workers,
                        args.suffix, args.sample_num)


    # define loss function (criterion)
    if args.dataset == 'VOC2012':
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    distilling = KDLoss().cuda(args.gpu)

    tea_model = None
    stu_model = None
    for g in range(args.start_gen, args.generations):
        # Create model
        if seed is not None: seed += 10
        stu_model = Generate_Model(args.dataset, args.arch, device_ids, args.train_layer, seed)

        # Create optimizer
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(stu_model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(stu_model.parameters(), args.lr, weight_decay=args.weight_decay)

        # For tensorboard
        logger_train = Logger('./BANlogs/{}_{}_{}_gen{}/train'.format(args.arch, args.dataset, args.suffix, g))
        logger_val = Logger('./BANlogs/{}_{}_{}_gen{}/val'.format(args.arch, args.dataset, args.suffix, g))

        # optionally resume from a checkpoint
        if g == args.start_gen and args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                tea_model = Generate_Model(args.dataset, args.arch, [args.gpu_teacher], args.train_layer, seed)
                checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(device_ids[0])))
                # args.start_epoch = checkpoint['epoch']
                # best_acc1 = checkpoint['best_acc1']
                tea_model.load_state_dict(checkpoint['state_dict'])
                tea_model.eval()
                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}) as Teacher!"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return
        else:
            args.start_epoch = 0

        training_one_net(stu_model, tea_model, criterion, distilling, optimizer, train_loader, val_loader,
                         args.epochs, g, logger_train, logger_val, args.start_epoch)
        del tea_model
        tea_model = stu_model
        tea_model.features = tea_model.features.module.cuda(args.gpu_teacher)
        tea_model.cuda(args.gpu_teacher)
        tea_model.eval()




def training_one_net(stu_model, tea_model, criterion, distilling, optimizer, train_loader, val_loader, epochs, gen, logger_train, logger_val,
                     start_epoch=0):
    global args, logspace_lr
    best_acc1 = 0
    for epoch in range(start_epoch, epochs):
        if args.logspace!=0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]
        else:
            adjust_learning_rate(optimizer, epoch)
        args.lambd = logspace_lambd[epoch]

        # train for one epoch
        train(train_loader, stu_model, tea_model, criterion, distilling, optimizer, epoch, gen, logger_train)

        # evaluate on validation set
        acc1 = validate(val_loader, stu_model, tea_model, criterion, distilling, epoch, logger_val)

        # remember best acc@1 and save checkpoint
        save_dir = 'checkpoint_{}_{}_{}_gen{}.pth.tar'.format(args.dataset, args.arch, args.suffix, gen)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if (epoch+1)%args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': stu_model.state_dict(),
                'best_acc1': best_acc1,
                'acc1': acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_dir)


def train(train_loader, stu_model, tea_model, criterion, distilling, optimizer, epoch, gen, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hard = AverageMeter(); losses_soft = AverageMeter();
    top0 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    stu_model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output

        output = stu_model(input)
        loss_hard = criterion(output, target)
        if tea_model is not None:
            with torch.no_grad():
                soft_tar = tea_model(input.cuda(args.gpu_teacher, non_blocking=True)).cuda(args.gpu, non_blocking=True)
            loss_soft = distilling(output, soft_tar)
            loss = loss_hard + args.lambd * loss_soft
        else:
            loss = loss_hard
        # measure accuracy and record loss
        if args.dataset == 'CUB':
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
        elif args.dataset == 'VOC2012':
            acc = accuracy_VOC2012(output, target)
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_hard.update(loss_hard.item(), input.size(0))
        if tea_model is not None:
            losses_soft.update(loss_soft.item(), input.size(0))
        if args.dataset == 'VOC2012':
            top0.update(acc[0], input.size(0))
        else:
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.dataset == 'VOC2012':
            if i % args.print_freq == 0:
                print('Gen: [{0}/{1}]\t'
                      'Epoch: [{2}][{3}/{4}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@0 {top0.val:.3f} ({top0.avg:.3f})\t'.format(
                    gen, args.generations, epoch, i, len(train_loader),
                    batch_time=batch_time, data_time=data_time, loss=losses, top0=top0))

        else:
            if i % args.print_freq == 0:
                print('Gen: [{0}/{1}]'
                    'Epoch: [{2}][{3}/{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   gen, args.generations, epoch, i, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))

    if args.dataset == 'VOC2012':
        log_dict = {'Loss': losses.avg, 'Loss_hard': losses_hard.avg, 'Loss_soft': losses_soft.avg, 'top0_prec': top0.avg.item()}
    else:
        log_dict = {'Loss':losses.avg, 'Loss_hard': losses_hard.avg, 'Loss_soft': losses_soft.avg, 'top1_prec':top1.avg.item(),'top5_prec':top5.avg.item()}
    set_tensorboard(log_dict,  epoch, logger)


def validate(val_loader, stu_model, tea_model, criterion, distilling, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_hard = AverageMeter(); losses_soft = AverageMeter();
    top0 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    stu_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = stu_model(input)
            loss_hard = criterion(output, target)
            if tea_model is not None:
                soft_tar = tea_model(input.cuda(args.gpu_teacher, non_blocking=True)).cuda(args.gpu, non_blocking=True)
                loss_soft = distilling(output, soft_tar)
                loss = loss_hard + args.lambd * loss_soft
            else:
                loss = loss_hard

            # measure accuracy and record loss
            if args.dataset == 'CUB':
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            elif args.dataset == 'VOC2012':
                acc = accuracy_VOC2012(output, target)
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            losses_hard.update(loss_hard.item(), input.size(0))
            if tea_model is not None:
                losses_soft.update(loss_soft.item(), input.size(0))
            if args.dataset == 'VOC2012':
                top0.update(acc[0], input.size(0))
            else:
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if args.dataset == 'VOC2012':
                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@0 {top0.val:.3f} ({top0.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top0=top0))
            else:
                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        if args.dataset == 'VOC2012':
            print(' * Acc@0 {top0.avg:.3f}'
                  .format(top0=top0))
        else:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        if args.dataset == 'VOC2012':
            log_dict = {'Loss': losses.avg, 'Loss_hard': losses_hard.avg, 'Loss_soft': losses_soft.avg, 'top0_prec': top0.avg.item()}
            set_tensorboard(log_dict, epoch, logger)
        else:
            log_dict = {'Loss': losses.avg, 'Loss_hard': losses_hard.avg, 'Loss_soft': losses_soft.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
            set_tensorboard(log_dict, epoch, logger)
    if args.dataset == 'VOC2012':
        return top0.avg
    else:
        return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_'+filename[11:])

# For tensorboard
def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)

    return

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
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
def accuracy_VOC2012(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        accur = output.gt(0.).long().eq(target.long()).float().mean()
        res = []
        res.append(accur)
        return res
if __name__ == '__main__':
    main()

