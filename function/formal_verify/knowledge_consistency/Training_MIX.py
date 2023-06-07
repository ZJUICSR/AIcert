'''
Note! This code is only for training MIX320 dataset
mentioned in the Paper.

'''
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
from Models import Generate_Model


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--train_path', default='../CUB_200_2011/crop/train', help='../../ILSVRC2012/', type=str)
parser.add_argument('--val_path', default='../CUB_200_2011/crop/test', type=str, help='../ILSVRC2012_img_val')
parser.add_argument('--data_path', default='', type=str)
parser.add_argument('--sample_rate', default=0, type=float)
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn_ft')
parser.add_argument('--optim', default='SGD',type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--dataset', default='mix320', type=str)
parser.add_argument('--epoch_step', default=60, type=int)
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--logspace',action='store_true')
parser.add_argument('--sample_num',default='',type=str)
parser.add_argument('--decay_factor', default=0.3, type=float)
parser.add_argument('--device_ids', default='[0,1]', type=str)
parser.add_argument('--val_epoch', default=5, type=int)
parser.add_argument('--train_layer', default=34, type=int)

best_acc1 = 0

args = parser.parse_args()
device_ids = json.loads(args.device_ids)
print('parsed options:', vars(args))

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

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    model = Generate_Model(args.dataset, args.arch, device_ids,
                           args.train_layer, args.seed, args.pretrained)


    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda(device_ids[0])
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        args.gpu = device_ids[0]
        print('GPU used: '+args.device_ids)
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            model.cuda(device_ids[0])
        else:
            model = torch.nn.DataParallel(model).cuda(device_ids[0])

    # define loss function (criterion) and optimizer
    if args.dataset == 'VOC2012':
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # logger_train = Logger('./logs/ILSVRC_sample_rate_{}{}/train'.format(args.sample_rate,args.suffix))
    # logger_val = Logger('./logs/ILSVRC_sample_rate_{}{}/val'.format(args.sample_rate, args.suffix))
    logger_train = Logger('./logs/{}_{}_{}/train'.format(args.dataset,args.arch,args.suffix))
    logger_val = Logger('./logs/{}_{}_{}/val'.format(args.dataset,args.arch, args.suffix))
    logger_val_dog = Logger('./logs/{}_{}_{}/val_dog'.format(args.dataset,args.arch, args.suffix))
    logger_val_cub = Logger('./logs/{}_{}_{}/val_cub'.format(args.dataset,args.arch, args.suffix))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(device_ids[0])))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr)-2, args.epochs)
    
    if args.dataset == 'mix320':
        # Data loading code
        traindir = "../mixedData/train/"
        valdir_dog = "../DOG120/crop/val/"
        valdir_cub = "../CUB_200_2011/crop/test/"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)
        
        val_dataset_dog = datasets.ImageFolder(valdir_dog, transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))
        
        val_dataset_cub = datasets.ImageFolder(valdir_cub, transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))

        num_val_cub = len(val_dataset_cub)
        num_val_dog = len(val_dataset_dog)

        val_loader_dog = torch.utils.data.DataLoader(
            val_dataset_dog,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
        val_loader_cub = torch.utils.data.DataLoader(
            val_dataset_cub,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        print("No dataset!")
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    is_best = False
    acc1 = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.logspace:
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]
        else:
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger_train)

        if (epoch+1)%args.val_epoch == 0:
        # evaluate on validation set
            acc1_cub, acc5_cub = validate(val_loader_cub, model, criterion, epoch, logger_val_cub, 'cub')
            acc1_dog, acc5_dog = validate(val_loader_dog, model, criterion, epoch, logger_val_dog, 'dog')
            acc1 = (acc1_cub * num_val_cub + acc1_dog * num_val_dog) / (num_val_cub + num_val_dog)
            acc5 = (acc5_cub * num_val_cub + acc5_dog * num_val_dog) / (num_val_cub + num_val_dog)
            log_dict = {'top1_prec': acc1, 'top5_prec': acc5}
            set_tensorboard(log_dict, epoch, logger_val)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        # remember best acc@1 and save checkpoint
        save_dir = 'checkpoint_{}_{}_{}.pth.tar'.format(args.dataset, args.arch, args.suffix)

        if (epoch+1)%args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'acc1': acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_dir)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top0 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        if args.dataset == 'CUB':
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
        elif args.dataset == 'VOC2012':
            acc = accuracy_VOC2012(output, target)
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
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
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@0 {top0.val:.3f} ({top0.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top0=top0))

        else:
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    if args.dataset == 'VOC2012':
        log_dict = {'Loss': losses.avg, 'top0_prec': top0.avg.item()}
    else:
        log_dict = {'Loss':losses.avg, 'top1_prec':top1.avg.item(),'top5_prec':top5.avg.item()}
    set_tensorboard(log_dict,  epoch, logger)


def validate(val_loader, model, criterion, epoch, logger, data):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top0 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    offset = 0 if data == 'cub' else 200
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target + offset
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if args.dataset == 'CUB':
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            elif args.dataset == 'VOC2012':
                acc = accuracy_VOC2012(output, target)
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            if args.dataset == 'VOC2012':
                top0.update(acc[0], input.size(0))
            else:
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(data+'\tTest: [{0}/{1}]\t'
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
            print(f' * {data} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        if args.dataset == 'VOC2012':
            log_dict = {'Loss': losses.avg, 'top0_prec': top0.avg.item()}
            set_tensorboard(log_dict, epoch, logger)
        else:
            log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
            set_tensorboard(log_dict, epoch, logger)
    if args.dataset == 'VOC2012':
        return top0.avg
    else:
        return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}_{}_{}.pth.tar'.format(args.dataset,args.arch, args.suffix))

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
