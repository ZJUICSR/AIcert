'''
To accelerate the training process, while maintain a low RAM consumption,
We save each image's feature map as separated files.

'''
import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import Model_zoo as models
from Datasets import Generate_Dataloader
import copy

parser = argparse.ArgumentParser(description='Tracing')

parser.add_argument('--device_ids', default='[1,2]', type=str)
parser.add_argument('--arch', default='vgg16_bn', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--resume1', default='', type=str)
parser.add_argument('--resume2', default='', type=str)
parser.add_argument('--dataset', default='CUB200', type=str)
parser.add_argument('--conv_layer', default=40, type=int)
parser.add_argument('--topk', default='[1,3]', type=str)
parser.add_argument('--t_or_v', default='train',type=str)
parser.add_argument('--fc', action='store_true')
parser.add_argument('--sample_num', default='', type=str)
parser.add_argument('--upsample', action='store_true')
args = parser.parse_args()
print('parsed options:', vars(args))

args.sample_num = ''
device_ids = json.loads(args.device_ids)
topk = json.loads(args.topk)

cuda.empty_cache()
def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(device_ids[0])))
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module')>=0:
                state_dict[key.replace('module.','')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
              .format(resume, checkpoint['epoch'], checkpoint['acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    del checkpoint, state_dict

def ResBlock_beforeReLU(block, x): # Only for ResNet 50 101 152!!
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


def main():
    cudnn.benchmark = True
    if args.dataset.startswith("VOC"):
        net1 = models.__dict__[args.arch](num_classes=20).cuda(device_ids[0])
        net2 = models.__dict__[args.arch](num_classes=20).cuda(device_ids[1])
    elif args.dataset.startswith("CUB"):
        net1 = models.__dict__[args.arch](num_classes=200).cuda(device_ids[0])
        net2 = models.__dict__[args.arch](num_classes=200).cuda(device_ids[1])
    elif args.dataset.startswith("DOG"):
        net1 = models.__dict__[args.arch](num_classes=120).cuda(device_ids[0])
        net2 = models.__dict__[args.arch](num_classes=120).cuda(device_ids[1])
    elif args.dataset=='cifar10':
        net1 = models.__dict__[args.arch](num_classes=10).cuda(device_ids[0])
        net2 = models.__dict__[args.arch](num_classes=10).cuda(device_ids[1])
    elif args.dataset=='mnist':
        net1 = models.__dict__[args.arch](num_classes=10).cuda(device_ids[0])
        net2 = models.__dict__[args.arch](num_classes=10).cuda(device_ids[1])

    load_checkpoint(args.resume1, net1)
    load_checkpoint(args.resume2, net2)

    net1.eval()
    net2.eval()
    
    if args.arch.startswith("vgg16_bn"):
        channels = 512
        kernel_size = 28 if args.conv_layer<=30 else 14
    elif args.arch.startswith("alexnet"):
        channels = 256
        kernel_size = 13
    elif args.arch.startswith('resnet'):
        if args.arch.startswith('resnet18') or args.arch.startswith('resnet34'):
            if args.conv_layer == 3:
                channels = 256
                kernel_size = 14
            elif args.conv_layer == 4:
                channels = 512
                kernel_size = 7
        else:
            if args.conv_layer == 3:
                channels = 1024
                kernel_size = 14
            elif args.conv_layer == 4:
                channels = 2048
                kernel_size = 7
    
    sub_idx = []
    
    if args.fc:
        channels = 4096
        kernel_size = 1

    # Create dataloader
    train_loader, val_loader = \
        Generate_Dataloader(args.dataset, args.batch_size, 4, args.suffix, args.sample_num)

    os.makedirs("convOuts/{}_{}_{}/{}_L{}_{}/{}".format(args.dataset, args.sample_num, args.arch, args.dataset, args.conv_layer, args.suffix, args.t_or_v), exist_ok=True)
    save_dir = "convOuts/{}_{}_{}/{}_L{}_{}/{}/".format(args.dataset, args.sample_num, args.arch, args.dataset, args.conv_layer, args.suffix, args.t_or_v)

    data_len = len(train_loader.dataset) if args.sample_num=='' else len(sub_idx)

    
    tar = torch.zeros(data_len,dtype=torch.int)
    pred1 = torch.zeros((data_len,len(topk)), dtype=torch.int)
    convOut1 = torch.zeros((data_len,channels, kernel_size, kernel_size))
    convOut2 = torch.zeros((data_len, channels, kernel_size, kernel_size))
    pred2 = torch.zeros((data_len,len(topk)), dtype=torch.int)

    tars = torch.zeros(1,dtype=torch.int)
    pred1s = torch.zeros(len(topk), dtype=torch.int)
    convOut1s = torch.zeros(channels, kernel_size, kernel_size)
    convOut2s = torch.zeros(channels, kernel_size, kernel_size)
    pred2s = torch.zeros(len(topk), dtype=torch.int)


    accum_cnt = 0
    num_batches = data_len // args.batch_size
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            batch_size = target.size(0)
            convOutBs1, predBs1 = validate(input, target, net1, device_ids[0])
            convOutBs2, predBs2 = validate(input, target, net2, device_ids[1])
            tar[accum_cnt:(accum_cnt+batch_size)] = target
            pred1[accum_cnt:(accum_cnt+batch_size),:] = predBs1
            pred2[accum_cnt:(accum_cnt+batch_size),:] = predBs2
            convOut1[accum_cnt:(accum_cnt+batch_size),:,:,:] = convOutBs1.data.cpu()
            convOut2[accum_cnt:(accum_cnt+batch_size),:,:,:] = convOutBs2.data.cpu()
            accum_cnt += batch_size
            print("Batch: {}/{}".format(i,num_batches))


        for j in tqdm(range(data_len)):
            tars[:] = tar[j]
            pred1s[:] = pred1[j]; pred2s[:] = pred2[j]
            convOut1s[:] = convOut1[j]; convOut2s[:] = convOut2[j]
            save_dict = {
                'target': tars,
                'pred1': pred1s,
                'pred2': pred2s,
                'convOut1': convOut1s,
                'convOut2': convOut2s,
            }
            torch.save(save_dict, save_dir+'{:06d}.pkl'.format(j))




def validate(input, target, model, device_id):
    input = input.cuda(device_id, non_blocking=True)
    target = target.cuda(device_id, non_blocking=True)
    model.eval()
    with torch.no_grad():
        if args.arch.startswith("alexnet"):
            if not args.fc:
                x = model.features[:args.conv_layer + 1](input)
                convOut = copy.deepcopy(x.data)
                featureOut = model.features[args.conv_layer + 1:](x)
                featureOut = model.avgpool(featureOut)
                featureOut = featureOut.view(featureOut.size(0), 256 * 6 * 6)
                output = model.classifier(featureOut)
                pred = get_pred(output, target, topk)
                return convOut, pred
            else:
                featureOut = model.features(input)
                featureOut = model.avgpool(featureOut)
                featureOut = featureOut.view(featureOut.size(0), 256 * 6 * 6)
                x = model.classifier[:args.conv_layer+1](featureOut)
                fcOut = copy.deepcopy(x.data)
                outout = model.classifier[args.conv_layer+1:](x)
                pred = get_pred(outout, target, topk)
    #             print(fcOut.size())
                fcOut = fcOut.reshape([-1, 4096, 1, 1])
    #             print(fcOut.size())
                return fcOut, pred

        elif args.arch.startswith("vgg"):
            if not args.fc:
                x = model.features[:args.conv_layer + 1](input)
                convOut = copy.deepcopy(x.data)
                featureOut = model.features[args.conv_layer + 1:](x)
                featureOut = featureOut.reshape(featureOut.size(0), -1)
                output = model.classifier(featureOut)
                pred = get_pred(output, target, topk)
                return convOut, pred
            else:
                featureOut = model.features(input)
                featureOut = featureOut.reshape(featureOut.size(0), -1)
                x = model.classifier[:args.conv_layer+1](featureOut)
                fcOut = copy.deepcopy(x.data)
                outout = model.classifier[args.conv_layer+1:](x)
                pred = get_pred(outout, target, topk)
    #             print(fcOut.size())
                fcOut = fcOut.reshape([-1, 4096, 1, 1])
    #             print(fcOut.size())
                return fcOut, pred
        
        elif args.arch.startswith("resnet"):
            x = input
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3[:-1](x)
            x = ResBlock_beforeReLU(model.layer3[-1], x)
            b3_beforeR = copy.deepcopy(x.data)
            b3 = nn.ReLU(inplace=True)(x)
            b4 = model.layer4(b3)
            x = model.avgpool(b4)
            
            x = x.view(x.size(0), -1)
            output = model.fc(x)
            pred = get_pred(output, target, topk)
            if args.conv_layer == 3:
                return b3_beforeR, pred
            # elif args.conv_layer == 4:
            #     return b4, pred
            else:
                return None, None



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
    lr = args.lr * (0.1 ** (epoch // 30))
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

def get_pred(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = torch.zeros((len(topk),target.size(0)),dtype=torch.int)
        for i in range(len(topk)):
            res[i,:] = torch.sum(correct[:topk[i],:],0).data.cpu()
        return res.t()


if __name__ == '__main__':
    main()
