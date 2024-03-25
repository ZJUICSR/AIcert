# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 10:30
# @File    : train_target_models.py

import numpy as np
from tqdm import tqdm
import os
import json
from torch import nn, optim
import torchvision,torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import socket
from load_model import load_model


# SERVICE_NAME = socket.gethostname()
# if SERVICE_NAME == 'ubuntu02':
#     BASE_DIR = '/opt/data/user/gss/code/data'
# else:
#     BASE_DIR = '/data/user/gss/code/results'

MODEL_SAVE_PATH = os.path.dirname(__file__)


def calc_acc(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    train_acc = list()
    loop = tqdm(enumerate(dataloader), ncols=100, desc=f'calc acc', total=len(dataloader), colour='green')
    for batch_idx, data in loop:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        mini_out = model(images)
        _, pred = torch.max(mini_out, 1)
        acc = (pred.data == labels).float().mean()
        train_acc.append(float(acc))
        loop.set_postfix({"Acc": f'{np.array(train_acc).mean():.6f}'})

    torch.cuda.empty_cache()

    return np.array(train_acc).mean()


def train_single_epoch(model,
                       dataloader: DataLoader,
                       lr_scheduler,
                       loss_func,
                       optimizer,
                       epoch: int,
                       device='cuda'):
    model.train()
    criterion = loss_func
    train_loss = list()
    train_acc = list()
    loop = tqdm(enumerate(dataloader), ncols=100, desc=f'Train epoch {epoch}', total=len(dataloader), colour='blue')
    for batch_idx, data in loop:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        mini_out = model(images)
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


def get_detect_model(dataset_name, model_name, device='cuda'):
    models = {'ids2018': {'resnet18': ids2018_detect_model,
                          'cnn': ids2018_cnn_detect_model},
              'kdd99': {'resnet18': kdd99_detect_model,
                        'cnn': kdd99_cnn_detect_model}}
    return models[dataset_name][model_name](device=device)


def train(dataset_name,
          dataloader: DataLoader,
          epochs=20,
          device='cuda',
          retrain=False,
          model_name='resnet18'):
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_model_name = f'{dataset_name}_{model_name}.pt'
    not_retrain_model = os.path.exists(os.path.join(MODEL_SAVE_PATH, save_model_name)) and not retrain
    model = load_model()
    model.to(device)
    train_epoch = -1

    if not_retrain_model:
        print('load model.pt')
        model_info = torch.load(os.path.join(MODEL_SAVE_PATH, save_model_name))
        train_epoch = model_info['epoch']
        model_param = model_info['model']
        model.load_state_dict(model_param)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)
    best_epoch_acc = 0
    save_json_file = os.path.join(MODEL_SAVE_PATH, f"{dataset_name}_{model_name}.json")
    if not os.path.exists(save_json_file):
        train_info = dict()
    else:
        with open(save_json_file, 'r', encoding='utf-8') as f:
            train_info = json.load(f)

    for epoch_num in range(1, epochs + 1):
        if epoch_num <= train_epoch:
            continue
        epoch_acc, epoch_loss = train_single_epoch(model=model,
                                                   dataloader=dataloader,
                                                   lr_scheduler=lr_scheduler,
                                                   loss_func=criterion,
                                                   optimizer=optimizer,
                                                   epoch=epoch_num,
                                                   device=device)
        if epoch_acc > best_epoch_acc:
            best_epoch_acc = epoch_acc
            model_info = {"epoch": epoch_num, 'acc': epoch_acc, 'model': model.state_dict()}
            torch.save(model_info, os.path.join(MODEL_SAVE_PATH, save_model_name))
        train_info.update({f'epoch_{epoch_num}': {"acc": epoch_acc, 'loss': epoch_loss}})
        with open(save_json_file, 'w', encoding='utf-8') as f:
            json.dump(train_info, f, indent=4, ensure_ascii=False)
    return model

def train_resnet_model(device='cuda', dataset_name='ids2018', batch_size=128, model_name='resnet18'):
    train_dataset = torchvision.datasets.MNIST(root='./mnist/', download=True, train=True, transform=transforms.ToTensor())
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = torchvision.datasets.MNIST(root='./mnist/', download=True, train=False, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = train(dataset_name=dataset_name,
                  dataloader=dataloader,
                  epochs=30,
                  device=device,
                  retrain=True,
                  model_name=model_name)
    acc = calc_acc(model=model, dataloader=dataloader, device=device)
    print(f'model acc ={acc}')
    acc = calc_acc(model=model, dataloader=test_dataloader, device=device)
    print(f'test model acc ={acc}')


if __name__ == '__main__':
    train_resnet_model(dataset_name='mnist', batch_size=256, model_name='resnet18')

