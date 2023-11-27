# -*- coding: utf-8 -*-
# @Time    : 2023/4/28 14:57
# @File    : gen_adv.py

# -*- coding: utf-8 -*-
import os
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import ConcatDataset
from torchattacks import *
from tqdm import tqdm
import json
import numpy as np


seed = 12395

def read_json(file_name=''):
    if not os.path.exists(file_name):
        return dict()
    with open(file_name, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


def write_json(info, file_name=''):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    return

def attacks_dict(model, eps):
    attacks = {
        'fgsm': FGSM(model=model, eps=eps),
        'bim': BIM(model=model, eps=eps, alpha=2 / 255, steps=10),
        'rfgsm': RFGSM(model, eps=eps, alpha=2 / 255, steps=10),
        'ffgsm': FFGSM(model=model, eps=eps, alpha=2 / 255),
        'tifgsm': TIFGSM(model=model, eps=eps, alpha=2 / 255, steps=10, decay=0.0, kernel_name='gaussian',
                         len_kernel=15, nsig=3, resize_rate=0.9, diversity_prob=0.5),
        'mifgsm': MIFGSM(model, eps=eps, alpha=2 / 255, steps=10, decay=0.1),
        'difgsm': DIFGSM(model=model, eps=eps, alpha=2 / 255, steps=10, diversity_prob=0.5, resize_rate=0.9),
        # 'spsa': SPSA(model=model, eps=eps, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64),
        'cw': CW(model=model, c=1, lr=0.01, steps=20, kappa=0),
        'upgd': UPGD(model=model, eps=eps, alpha=2 / 255, steps=10),
        'pgd': PGD(model=model, eps=eps, alpha=2 / 225, steps=10, random_start=True),
        'tpgd': TPGD(model=model, eps=eps, alpha=2 / 255, steps=10),
        'pgdl2': PGDL2(model=model, eps=eps, alpha=0.2, steps=10),
        'sparsefool': SparseFool(model=model, steps=10, lam=3, overshoot=0.02),
        'deepfool': DeepFool(model=model, steps=10),

    }
    return attacks


def get_attack(method, model, eps):
    eps = eps / 255
    attacks = attacks_dict(model=model, eps=eps)
    if method.lower() not in attacks:
        return None
    return attacks[method.lower()]


class AtDataset(Dataset):
    def __init__(self, data=[], label=[]):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_adv_data(model, attack, dataloader, device='cuda', desc=''):
    x_adv, x, ori_y, adv_y, acc = None, None, None, None, 0
    correct = 0
    loop = tqdm(dataloader, total=len(dataloader), leave=True, desc=desc, ncols=100)
    acc_list = list()
    model.eval()

    for data, label in loop:
        data, label = data.to(device), label.to(device)
        adv = attack(data, label)
        outputs = model(adv)
        _, pre = torch.max(outputs.data, 1)
        correct += float((pre == label).sum()) / len(label)
        acc_list.append(float((pre == label).sum()) / len(label))
        loop.set_postfix({"Acc": f'{np.array(acc_list).mean():.6f}'})
        if x_adv is None:
            x_adv = adv.detach().to('cpu')
            x = data.detach().to('cpu')
            ori_y = label.detach().to('cpu')
            adv_y = pre.detach().to('cpu')
            continue
        x_adv = torch.cat((x_adv, adv.detach().to('cpu')), dim=0)
        x = torch.cat((x, data.detach().to('cpu')), dim=0)
        ori_y = torch.cat((ori_y, label.detach().to('cpu')), dim=0)
        adv_y = torch.cat((adv_y, pre.detach().to('cpu')), dim=0)

    acc = round(correct / len(dataloader), 6)
    return x_adv, x, ori_y, adv_y, acc


def generate_at_dataloader(attack_info, batch_size):
    adversarial_dataset = AtDataset(data=attack_info[0], label=attack_info[2])
    clean_dataset = AtDataset(data=attack_info[1], label=attack_info[2])
    dataset = ConcatDataset([adversarial_dataset, clean_dataset])

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    adv_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    adv_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return adv_train_loader, adv_test_loader, attack_info[-1]

def get_at_dataloader(method='fgsm', model=None, dataloader=None, eps=1, device='cuda', batch_size=128):
    torch.manual_seed(seed)
    attack = get_attack(method=method, model=model, eps=eps)
    attack_info = get_adv_data(model=model, attack=attack, dataloader=dataloader, device=device, desc=f'生成{method}对抗样本，eps={eps}')

    return generate_at_dataloader(attack_info, batch_size)


def calc_acc(model,  dataloader, device='cuda', desc=''):
    correct = 0
    model.eval()
    for data, label in tqdm(dataloader, total=len(dataloader), leave=True, desc=desc):
        data, label = data.to(device), label.to(device)
        outputs = model(data)
        _, pre = torch.max(outputs, 1)
        correct += float((pre == label).sum()) / len(label)

    acc = round(correct / len(dataloader), 6)
    return acc



if __name__ == '__main__':
    pass

