# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
# from resnet import ResNet18
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
def freeat_adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)



def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    return ((output.data.max(1)[1] == target.data).float().sum()) / len(target)


def get_model_names():
	return sorted(name for name in models.__dict__
    		if name.islower() and not name.startswith("__")
    		and callable(models.__dict__[name]))

def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\


def validate_pgd(test_loader, model, criterion, K, step):
    # Mean/Std for normalization
    for i, (input, target) in enumerate(test_loader):
        crop_size = len(input[0][0])
        break
    if crop_size == 32:
        args_mean = [0.485, 0.456, 0.406]
        args_std = [0.229, 0.224, 0.225]
        mean = torch.Tensor(np.array(args_mean)[:, np.newaxis, np.newaxis])
        mean = mean.expand(3,crop_size, crop_size).cuda()
        std = torch.Tensor(np.array(args_std)[:, np.newaxis, np.newaxis])
        std = std.expand(3, crop_size, crop_size).cuda()
    elif crop_size == 28:
        args_mean = [0.1307]
        args_std = [0.3081]
        mean = torch.Tensor(np.array(args_mean)[:, np.newaxis, np.newaxis])
        mean = mean.expand(1,crop_size, crop_size).cuda()
        std = torch.Tensor(np.array(args_std)[:, np.newaxis, np.newaxis])
        std = std.expand(1, crop_size, crop_size).cuda()
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    eps = clip_eps
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input-eps, input)
            input = torch.min(orig_input+eps, input)
            input.clamp_(0, 1.0)
        
        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('PGD Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg

def validate(test_loader, model, criterion):
    # Mean/Std for normalization
    for i, (input, target) in enumerate(test_loader):
        crop_size = len(input[0][0])
        break
    if crop_size == 32 and len(input[0]) == 3:
        args_mean = [0.485, 0.456, 0.406]
        args_std = [0.229, 0.224, 0.225]
        mean = torch.Tensor(np.array(args_mean)[:, np.newaxis, np.newaxis])
        mean = mean.expand(3,crop_size, crop_size).cuda()
        std = torch.Tensor(np.array(args_std)[:, np.newaxis, np.newaxis])
        std = std.expand(3, crop_size, crop_size).cuda()
    else:
        args_mean = [0.1307]
        args_std = [0.3081]
        mean = torch.Tensor(np.array(args_mean)[:, np.newaxis, np.newaxis])
        mean = mean.expand(1,crop_size, crop_size).cuda()
        std = torch.Tensor(np.array(args_std)[:, np.newaxis, np.newaxis])
        std = std.expand(1, crop_size, crop_size).cuda()
    
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    return top1.avg

print_freq = 10
n_repeats = 4
# crop_size = 32
max_color_value = 255.0
# args_mean = [0.485, 0.456, 0.406]
# args_std = [0.229, 0.224, 0.225]
clip_eps = 4.0
fgsm_step = 4.0
fgsm_step /= max_color_value
clip_eps /= max_color_value
 
def freeat_train(train_loader, model, criterion, optimizer, epoch, global_noise_data):
    for i, (input, target) in enumerate(train_loader):
        crop_size = len(input[0][0])
        break
    if crop_size == 32 and len(input[0]) == 3:
        args_mean = [0.485, 0.456, 0.406]
        args_std = [0.229, 0.224, 0.225]
        mean = torch.Tensor(np.array(args_mean)[:, np.newaxis, np.newaxis])
        mean = mean.expand(3,crop_size, crop_size).cuda()
        std = torch.Tensor(np.array(args_std)[:, np.newaxis, np.newaxis])
        std = std.expand(3, crop_size, crop_size).cuda()
    else:
        args_mean = [0.1307]
        args_std = [0.3081]
        mean = torch.Tensor(np.array(args_mean)[:, np.newaxis, np.newaxis])
        mean = mean.expand(1,crop_size, crop_size).cuda()
        std = torch.Tensor(np.array(args_std)[:, np.newaxis, np.newaxis])
        std = std.expand(1, crop_size, crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        for j in range(n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)
            
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-clip_eps, clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
                sys.stdout.flush()

if __name__ == '__main__':
    # main()
    pass
