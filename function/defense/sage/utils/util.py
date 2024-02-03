from __future__ import print_function
import torch
import os
from torch.utils.data import random_split, Dataset

class AverageMeter(object):
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


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def load_pretrained_model(model, pretrained_dict, wfc=True):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if wfc:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and ('fc' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s

#num(fx n-1 < fx n)<0.75*(n - (n-1))
#lr = lr and fmax x =  
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate1(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def adjust_learning_rate(optimizer, epoch, prev_lr, curr_lr, curr_min_loss, prev_min_loss, counts):
    pre_lr = prev_lr
    if epoch % 25 == 0:
        pre_lr = curr_lr
        if counts < 0.75 * 25 or (prev_lr == curr_lr and curr_min_loss >= prev_min_loss):
            curr_lr = curr_lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
            counts = 0    
        prev_min_loss = curr_min_loss
    print('epoch: {}  lr: {:.4f}'.format(epoch, curr_lr))
    return pre_lr, curr_lr, prev_min_loss, counts

def save_checkpoint(state, fdir, model_name):
    filepath = os.path.join(fdir, model_name + '.ckpt')
    torch.save(state, filepath)
    print('[info] save best model')

class DatasetFull(Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=1)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        test_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - test_size
        test_dataset, drop_dataset = random_split(
            full_dataset, [test_size, drop_size])
        print('test_size:', len(test_dataset),
              'drop_size:', len(drop_dataset))

        return test_dataset

class DatasetPart(Dataset):
    def __init__(self, ratio = 0.2, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(
            full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset),
              'drop_size:', len(drop_dataset))

        return train_dataset