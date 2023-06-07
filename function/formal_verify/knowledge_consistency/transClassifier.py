import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.cuda as cuda
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Datasets import Generate_Dataloader
import numpy as np
from logger import Logger
import Model_zoo as models

import os
from torch.nn.modules.loss import _Loss


parser = argparse.ArgumentParser(description='Tracing')
parser.add_argument('--arch', default='vgg16_bn', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--net_A', default='../New_Models_Re/checkpoint_CUB200_vgg16_bn_lr-2_sd0_itr300.pth.tar', type=str)
parser.add_argument('--net_B', default='../New_Models_Re/checkpoint_CUB200_vgg16_bn_lr-2_sd5_itr300.pth.tar', type=str)
parser.add_argument('--resume_Ys', default='model_checkpoints/CUB200_vgg16_bn/checkpoint_L30__a0.1_lr-4.pth.tar',
                    type=str)
parser.add_argument('--dataset', default='CUB200', type=str)
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=3, type=int,
                    help='GPU id to use.')
parser.add_argument('--resumePath', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--conv_layer', default=30, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--topk', default='[1,3]', type=str)
parser.add_argument('--suffix', default='test', type=str)
# parser.add_argument('--suffix', default='trainData_0.1_release_nosub', type=str)
parser.add_argument('--in_mode', default='[1,1,1]', type=str)
# parser.add_argument('--sub_sampler',
#                     default='/home/data/lilongfei/FeatureFactorization/sub_sampler_VOC2012_crop_10par_C.npy',
#                     type=str)
parser.add_argument('--logspace', default=2, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--save_epoch', default=100, type=int)
parser.add_argument('--val_epoch', default=10, type=int)
# parser.add_argument('--save_per_epoch', default=False, type=bool)
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--sample_num', default='', type=str)

args = opt = parser.parse_args()
if args.arch.startswith("vgg"):
    convName = {"conv4_1": 24, "conv4_2": 27, "conv4_3": 30, "conv5_1": 34, "conv5_2": 37, "conv5_3": 40, "FC6": 0,
                "FC7": 3}
elif args.arch.startswith("alexnet"):
    convName = {"conv4": 8, "conv5": 10, "FC6": 1, "FC7": 4}
elif args.arch.startswith("resnet"):
    convName = {"layer3": 3}
for x, y in convName.items():
    if y == args.conv_layer:
        conv = x
print("extract feature maps from {}\n".format(conv))

print('parsed options:', vars(opt))

topk = json.loads(opt.topk)
in_mode = json.loads(opt.in_mode)

cuda.empty_cache()


def ResBlock_beforeReLU(block, x):  # Only for ResNet 50 101 152!!
    identity = x

    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)

    out = block.conv2(out)
    out = block.bn2(out)
    if block.__class__.__name__ != 'BasicBlock':
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(x)

    out += identity
    return out


class img_to_feature(nn.Module):
    def __init__(self, fcn, f_trans=None):  # fcn: vgg16_bn[:41]; f_trans: linearTest
        super(img_to_feature, self).__init__()
        self.fcn = fcn
        self.trans = f_trans
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.fcn(x)
            if self.trans:
                out, out_n = self.trans.val_batch(x)
            return out, out_n


class TransClassifier(nn.Module):
    def __init__(self, ori_net, layer=0):
        super(TransClassifier, self).__init__()
        if args.arch.startswith("alexnet"):
            self.features = ori_net.features[layer:]
            self.classifier = ori_net.classifier
            self.avgpool = ori_net.avgpool
        elif args.arch.startswith("vgg"):
            self.features = ori_net.features[layer:]
            self.classifier = ori_net.classifier
        elif args.arch.startswith("resnet"):
            self.layer3 = ori_net.layer3
            self.layer4 = ori_net.layer4
            self.avgpool = ori_net.avgpool
            self.fc = ori_net.fc
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        if args.arch.startswith("alexnet"):
            out = self.features(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), 256 * 6 * 6)
            out = self.classifier(out)
        elif args.arch.startswith("vgg"):
            out = self.features(x)
            out = out.view(out.size(0), -1).cuda(opt.gpu)
            out = self.classifier(out)
        elif args.arch.startswith("resnet"):
            x = nn.ReLU(inplace=True)(x)
            out = self.layer4(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out


class head_resnet(nn.Module):
    def __init__(self, ori_net):
        super(head_resnet, self).__init__()
        self.conv1 = ori_net.conv1
        self.bn1 = ori_net.bn1
        self.relu = ori_net.relu
        self.maxpool = ori_net.maxpool
        self.layer1 = ori_net.layer1
        self.layer2 = ori_net.layer2
        self.layer3 = ori_net.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[:-1](x)
        b3_beforeR = ResBlock_beforeReLU(self.layer3[-1], x)
        return b3_beforeR


def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(opt.gpu)))
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module') != -1:
                state_dict[key.replace('module.', '')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        if 'best_acc1' in checkpoint:
            print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
                  .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    del checkpoint, state_dict


class M_Dataset(torch.utils.data.Dataset):
    def __init__(self, in_mode, suffix, channels=3):
        super(M_Dataset, self).__init__()
        self.target = torch.load('./M_Output/M_Output_target_{}.pkl'.format(suffix))
        if sum(in_mode) == channels:
            convOut = torch.load('./M_Output/M_Output_Y_sum_{}.pkl'.format(suffix))
            print("load ./M_Output/M_Output_Y_sum_{}.pkl".format(suffix))
        else:
            k = 0
            for i in range(channels):
                if in_mode[i] == 1:
                    tmp = torch.load('./M_Output/M_Output_Y{}_{}.pkl'.format(i, suffix))
                    print("load ./M_Output/M_Output_Y{}_{}.pkl".format(i, suffix))
                    if k == 0:
                        convOut = torch.zeros_like(tmp)
                    convOut += tmp
                    k += 1
                    del tmp
        self.convOut = convOut

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        convOut = self.convOut[idx]
        target = self.target[idx]

        return (convOut, target)


def main():
    if args.dataset.startswith("VOC"):
        netA = models.__dict__[args.arch](num_classes=20).cuda(args.gpu)
        netB = models.__dict__[args.arch](num_classes=20).cuda(args.gpu)
    elif args.dataset.startswith("CUB"):
        netA = models.__dict__[args.arch](num_classes=200).cuda(args.gpu)
        netB = models.__dict__[args.arch](num_classes=200).cuda(args.gpu)
    elif args.dataset.startswith("DOG"):
        netA = models.__dict__[args.arch](num_classes=120).cuda(args.gpu)
        netB = models.__dict__[args.arch](num_classes=120).cuda(args.gpu)
    elif args.dataset.startswith("Mix"):
        netA = models.__dict__[args.arch](num_classes=320).cuda(args.gpu)
        netB = models.__dict__[args.arch](num_classes=320).cuda(args.gpu)

    # load vgg model
    print("load model_vgg......")
    load_checkpoint(opt.net_A, netA)
    load_checkpoint(opt.net_B, netB)

    if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
        vggB_part = TransClassifier(netB, args.conv_layer + 1)
        vggA_part = netA.features[:args.conv_layer + 1]
    elif args.arch.startswith("resnet"):
        vggB_part = TransClassifier(netB)
        vggA_part = head_resnet(netA)

    # load 3 layers model
    print("load model_Ys......")
    #######################################################
    if args.arch.startswith("vgg"):
        if args.conv_layer <= 30:
            input_size = output_size = torch.zeros((512, 28, 28)).shape
        else:
            input_size = output_size = torch.zeros((512, 14, 14)).shape
    elif args.arch.startswith("alexnet"):
        input_size = output_size = torch.zeros((256, 13, 13)).shape
    elif args.arch.startswith("resnet"):
        if args.arch.startswith('resnet18') or args.arch.startswith('resnet34'):
            input_size = output_size = torch.zeros((256, 14, 14)).shape
        else:
            input_size = output_size = torch.zeros((1024, 14, 14)).shape
    #######################################################

    model_Ys = models.LinearTester(input_size, output_size, gpu_id=args.gpu, fix_p=True, bn=False, instance_bn=True)
    if args.resume_Ys == "":
        resume_Ys = "./model_checkpoints/VOC2012_crop/checkpoint_L{}_{}_3.0.pth.tar".format(args.conv_layer,
                                                                                            args.sample_num)
    else:
        resume_Ys = args.resume_Ys

    load_checkpoint(resume_Ys, model_Ys)

    catA = img_to_feature(vggA_part, model_Ys)

    # Create dataloader
    train_loader, val_loader = \
        Generate_Dataloader(args.dataset, args.batch_size, args.workers,
                        args.suffix, args.sample_num)

    if args.gpu is not None:
        catA = catA.cuda(args.gpu)
        vggB_part = vggB_part.cuda(args.gpu)
    else:
        print("error: gpu not assigned")

    if not os.path.exists("./logs_convs_vgg2trans/{}_{}_{}".format(opt.dataset, opt.sample_num, opt.arch)):
        os.makedirs("./logs_convs_vgg2trans/{}_{}_{}".format(opt.dataset, opt.sample_num, opt.arch), exist_ok=True)
    logger_train = Logger(
        './logs_convs_vgg2trans/{}_{}_{}/L{}_{}_{}_{}_{}/train'.format(opt.dataset, opt.sample_num, opt.arch,
                                                                       opt.conv_layer, opt.dataset, in_mode, opt.lr,
                                                                       opt.suffix))
    logger_val = Logger(
        './logs_convs_vgg2trans/{}_{}_{}/L{}_{}_{}_{}_{}/val'.format(opt.dataset, opt.sample_num, opt.arch,
                                                                     opt.conv_layer, opt.dataset, in_mode, opt.lr,
                                                                     opt.suffix))

    # define loss function (criterion) and optimizer
    if args.dataset == 'VOC2012':
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(vggB_part.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(vggB_part.parameters(), args.lr, weight_decay=args.weight_decay)

    if opt.resume:
        if os.path.isfile(opt.resumePath):
            print("=> loading checkpoint '{}'".format(opt.resumePath))
            checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(opt.gpu)))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            vggB_part.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = logspace_lr[epoch]
        # train for one epoch
        train(train_loader, vggB_part, catA, criterion, optimizer, epoch, logger_train)
        if epoch % args.val_epoch == 9:
            acc1 = validate(val_loader, vggB_part, catA, criterion, epoch, logger_val)

            # # remember best acc@1 and save checkpoint
            # # evaluate on validation sets
            #
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not os.path.exists("./check/{}_{}_{}/".format(opt.dataset, opt.sample_num, opt.arch)):
                os.mkdir("./check/{}_{}_{}/".format(opt.dataset, opt.sample_num, opt.arch))
            save_dir = save_dir_itr = './check/{}_{}_{}/checkpoint_{}_{}_L{}.pth.tar'.format(args.dataset,
                                                                                             args.sample_num, opt.arch,
                                                                                             args.in_mode, args.suffix,
                                                                                             args.conv_layer)
            if epoch > 0 and epoch % args.save_epoch == 99:
                save_checkpoint(
                    {'epoch': epoch + 1,
                     'arch': args.arch,
                     'state_dict': vggB_part.state_dict(),
                     'best_acc1': best_acc1,
                     'optimizer': optimizer.state_dict(), },
                    is_best,
                    save_dir)
    return


def train(train_loader, vggB_part, catA, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top0 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    catA.eval()
    vggB_part.train()

    if args.dataset == "VOC2012":
        Targets = torch.zeros((args.batch_size, 20))
    else:
        Targets = torch.zeros(args.batch_size)

    end = time.time()
    for i, (datas, target) in enumerate(train_loader):
        if args.gpu is not None:
            datas = datas.cuda(args.gpu, non_blocking=True)
        # output Ys
        output, output_n = catA(datas)

        if args.dataset == "Mix_DOG120":
            target = target + 200
        # start to finetune
        #  measure Ys loading time
        data_time.update(time.time() - end)
        input = output_n[0] * in_mode[0] + output_n[1] * in_mode[1] + output_n[2] * in_mode[2]
        # print(input.grad_fn)
        # print(input[8])
        # print(output_n[0][8])
        # print(output_n[1][8])
        # print(output_n[2][8])
        # exit()
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)
        output = vggB_part(input)

        if args.dataset == "VOC2012":
            loss = criterion(output, target.float())
        else:
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
        log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)


def validate(val_loader, vggB_part, catA, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top0 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    catA.eval()
    vggB_part.eval()

    with torch.no_grad():
        if args.dataset == "VOC2012":
            Targets = torch.zeros((args.batch_size, 20))
        else:
            Targets = torch.zeros(args.batch_size)

        end = time.time()
        for i, (datas, target) in enumerate(val_loader):
            if args.gpu is not None:
                datas = datas.cuda(args.gpu, non_blocking=True)
            # output Ys
            output, output_n = catA(datas)
            if args.dataset == "Mix_DOG120":
                target = target + 200
            input = output_n[0] * in_mode[0] + output_n[1] * in_mode[1] + output_n[2] * in_mode[2]

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.long().cuda(args.gpu, non_blocking=True)

            # compute output
            output = vggB_part(input)
            if args.dataset == "VOC2012":
                loss = criterion(output, target.float())
            else:
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
            log_dict = {'Loss': losses.avg, 'top0_prec': top0.avg.item()}
            set_tensorboard(log_dict, epoch, logger)
        else:
            log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
            set_tensorboard(log_dict, epoch, logger)
        if args.dataset == 'VOC2012':
            return top0.avg
        else:
            return top1.avg


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        './check/checkpoint_{}_{}_{}_{}_best.pth.tar'.format(args.in_mode, args.lr, args.suffix,
                                                                             args.conv_layer))


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
