# -*- coding: utf-8 -*-
import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
sys.path.append("../..")
from function.ensemble.paca_detect.net import TwoStraeamSrncovet
from tqdm import tqdm
import numpy as np
from function.ensemble.models.load_model import load_model
import json
from function.ensemble.attack.gen_adv import get_adv_dataloader
from function.ensemble.paca_detect.eval_model import ModelScore
from function.ensemble.paca_detect.json_op import save_as_json_file
import socket


MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adv_detect', 'detect_models')
RESULT_SAVE_PATH = MODEL_SAVE_PATH

for path in [MODEL_SAVE_PATH, RESULT_SAVE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


def train_single_epoch(detect_model,
                       target_model,
                       dataloader: DataLoader,
                       lr_scheduler,
                       loss_func,
                       optimizer,
                       epoch: int,
                       device='cuda'):
    detect_model.train()
    lr_scheduler.step(epoch=epoch)
    criterion = loss_func
    train_loss = list()
    train_acc = list()

    loop = tqdm(enumerate(dataloader), ncols=100, desc=f'Train epoch {epoch}', total=len(dataloader), colour='blue')
    for batch_idx, data in loop:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        images = torch.autograd.Variable(images, requires_grad=True)
        y_logit = target_model(images)
        _, pred_class = torch.max(y_logit, 1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, images)[0]
        gradient = torch.abs(gradient).detach().to(device)
        mini_out = detect_model(images, gradient)
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

    return np.array(train_acc).mean(), np.array(train_loss).mean()

def train_adv_detect_model(model_name,
                           target_model,
                           dataloader: DataLoader,
                           in_channel=1,
                           epoches=20,
                           device='cuda',
                           retrain=False,
                           save=False):
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    detect_model = TwoStraeamSrncovet(in_channel=in_channel)
    detect_model.to(device)

    train_epoch = -1
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, model_name)) and not retrain:
        model_info = torch.load(os.path.join(MODEL_SAVE_PATH, model_name))
        train_epoch = model_info['epoch']
        model_param = model_info['model']
        detect_model.load_state_dict(model_param)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(detect_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)
    best_epoch_acc = 0

    json_save_file = os.path.join(MODEL_SAVE_PATH, f"{model_name.split('.')[0]}.json")
    if not os.path.exists(json_save_file):
        train_info = dict()
    else:
        with open(json_save_file, 'r', encoding='utf-8') as f:
            train_info = json.load(f)

    for epoch_num in range(1, epoches + 1):
        if epoch_num <= train_epoch:
            continue
        epoch_acc, epoch_loss = train_single_epoch(detect_model=detect_model,
                                                   target_model=target_model,
                                                   dataloader=dataloader,
                                                   lr_scheduler=lr_scheduler,
                                                   loss_func=criterion,
                                                   optimizer=optimizer,
                                                   epoch=epoch_num,
                                                   device=device)
        if epoch_acc > best_epoch_acc:
            best_epoch_acc = epoch_acc
            model_info = {"epoch": epoch_num, 'acc': epoch_acc, 'model': detect_model.state_dict()}
            if save:
                torch.save(model_info, os.path.join(MODEL_SAVE_PATH, model_name))
        train_info.update({f'epoch_{epoch_num}': {"acc": epoch_acc, 'loss': epoch_loss}})
        if save:
            with open(json_save_file, 'w', encoding='utf-8') as f:
                json.dump(train_info, f, indent=4, ensure_ascii=False)
        if epoch_acc > 0.95:
            break
    return {'detect_model': detect_model, 'detect_acc': epoch_acc, 'attack_asr': 1- epoch_acc}

def train_detect_model(victim_model=None, dataloader=None, in_channel=1, method='fgsm',
                       attackparam = {'eps':1}, epoches=15, device='cuda', batch_size=128, gen_adv=True):
    if 'eps' in attackparam:
        eps = attackparam['eps']
    else:
        eps = 1
    model = victim_model
    model.to(device)
    if gen_adv:
        train_dataloader, test_dataloader = get_adv_dataloader(method=method, model=model, dataloader=dataloader, attackparam = attackparam, device=device, batch_size=batch_size)
    else:
        train_dataloader = dataloader
    detect_info = train_adv_detect_model(model_name=f'paca_{method}_eps_{eps}_255.pt',
                                         target_model=model,
                                         dataloader=train_dataloader,
                                         in_channel=in_channel,
                                         device=device,
                                         epoches=epoches,
                                         retrain=True)
    return detect_info


if __name__ == '__main__':
    from function.ensemble.datasets.mnist import mnist_dataloader
    methods = ['fgsm', 'bim', 'rfgsm', 'cw', 'pgd', 'tpgd', 'mi-fgsm', 'autopgd', 'square', 'deepfool', 'difgsm']
    method = 'FGSM'
    device ='cuda'
    eps = 1
    model = load_model()
    model.to(device)
    dataloader, _ = mnist_dataloader()
    detect_info = train_detect_model(victim_model=model, dataloader=dataloader, in_channel=1,
                                     method=method, eps=eps, epoches=15, device=device, batch_size=128)
    print(f'detect_acc={detect_info["detect_acc"]}')
    print(f'attack_asr={detect_info["attack_asr"]}')
    print(detect_info['attack_asr'])
