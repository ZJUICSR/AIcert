# -*- coding: utf-8 -*-
# @Time    : 2023/4/28 14:57
# @File    : gen_adv.py

# -*- coding: utf-8 -*-
import os, sys
# sys.path.append("..")
sys.path.append("../../..")
from function.attack import run_get_adv_data
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import ConcatDataset
from torchattacks import *
from tqdm import tqdm
import json
import numpy as np
from .DeepFool_My import DeepFoolMy

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

def get_attack(method, model, eps):
    eps = eps / 255
    attacks = {
                'fgsm': FGSM(model=model, eps=eps),
                'bim': BIM(model=model, eps=eps, alpha=2/255, steps=10),
                'rfgsm': RFGSM(model, eps=eps, alpha=2/255, steps=10),
                'ffgsm': FFGSM(model=model, eps=eps, alpha=2/255),
                'tifgsm': TIFGSM(model=model, eps=eps, alpha=2/255, steps=10, decay=0.0, kernel_name='gaussian', len_kernel=15, nsig=3, resize_rate=0.9, diversity_prob=0.5),
                'nifgsm': NIFGSM(model=model, eps=eps, alpha=2/255, steps=10, decay=1.0),
                'sinfgsm': SINIFGSM(model=model, eps=eps, alpha=2/255, steps=10, decay=1.0, m=5),
                'vmifgsm': VMIFGSM(model=model, eps=eps, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2),
                'vnifgsm': VNIFGSM(model=model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2),
                'mifgsm': MIFGSM(model, eps=eps, alpha=2 / 255, steps=10, decay=0.1),
                'difgsm': DIFGSM(model=model, eps=eps, alpha=2/255, steps=10, diversity_prob=0.5, resize_rate=0.9),
                'spsa': SPSA(model=model, eps=eps, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64),
                'cw': CW(model=model, c=1, lr=0.01, steps=20, kappa=0),
                'upgd': UPGD(model=model, eps=eps, alpha=2/255, steps=10),
                'pgd': PGD(model=model, eps=eps, alpha=2/225, steps=10, random_start=True),
                'tpgd': TPGD(model=model, eps=eps, alpha=2/255, steps=10),
                'pgdl2': PGDL2(model=model, eps=eps, alpha=0.2, steps=10),
                'pgdrsl2': PGDRSL2(model=model, eps=eps, alpha=0.2, steps=10),
                'sparsefool': SparseFool(model=model, steps=10, lam=3, overshoot=0.02),
                'autopgd': APGD(model=model, eps=eps, steps=10, eot_iter=1, n_restarts=1, loss='ce'),
                'onepixel': OnePixel(model=model, pixels=1, steps=10, popsize=10, inf_batch=128),
                'square': Square(model=model, eps=eps, n_queries=10, n_restarts=1, loss='ce'),
                'pixle': Pixle(model=model, x_dimensions=(2, 10), y_dimensions=(2, 10), pixel_mapping='random', restarts=20, max_iterations=10,),
                'deepfool': DeepFoolMy(model=model, steps=10)
                }
    if method.lower() not in attacks:
        return None
    return attacks[method.lower()]


class AdversarialDataset(Dataset):
    def __init__(self, data=[], label=0):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        return img, self.label


class AtDataset(Dataset):
    def __init__(self, data=[], label=[]):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class CafdAdversarialDataset(Dataset):
    def __init__(self, clean_x=None, adv_x=None, label=None):
        self.clean_x = clean_x
        self.adv_x = adv_x
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.clean_x[idx], self.adv_x[idx], self.label[idx]


class Cifar10Datasets(Dataset):
    def __init__(self, data={}):
        super(Cifar10Datasets, self).__init__()
        self.data = data
        self._x = data['x']
        self._y = data['y'] if 'y' in data else data['ori_y']

    def __len__(self):
        return len(self._y)

    def __getitem__(self, item):
        return self._x[item], self._y[item]


def remove_nan_in_tensor(ori_data: torch.Tensor):
    data = ori_data.detach().cpu()
    split_data = ori_data.chunk(data.size(0), dim=0)
    result = None
    for d in split_data:
        if d.isnan().sum() > 64:
            continue
        if result is None:
            result = d
            continue
        result = torch.cat([result, d], dim=0)
    return result


def get_adv_data(model, attack, dataloader, device='cuda', desc=''):
    x_adv, x, ori_y, adv_y, acc = None, None, None, None, 0
    correct = 0
    loop = tqdm(dataloader, total=len(dataloader), leave=True, desc=desc, ncols=100)
    acc_list = list()
    model.eval()
    num = 0
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


def prepare_data(dataloader):
    x, y = None, None
    for data, label in tqdm(dataloader, desc='Prepare Data', ncols=100):
        if x is None:
            x = data.cpu()
            y = label.cpu()
            continue
        x = torch.cat((x, data.cpu()), dim=0)
        y = torch.cat((y, label.cpu()), dim=0)
    return x, y


def generate_paca_dataloader(attack_info, batch_size):
    adversarial_dataset = AdversarialDataset(data=attack_info[0], label=1)
    clean_dataset = AdversarialDataset(data=attack_info[1], label=0)
    dataset = ConcatDataset([adversarial_dataset, clean_dataset])

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    adv_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    adv_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return adv_train_loader, adv_test_loader

def get_adv_dataloader(method='fgsm', model=None, dataloader=None, attackparam = {'eps':1}, device='cuda', batch_size=128):
    torch.manual_seed(seed)
    attack_info = run_get_adv_data(dataset_name='mnist', model = model, dataloader=dataloader, device=device, method=method, attackparam=attackparam)
    # AttackObj = EvasionAttacker(dataset="mnist", device=device, datanormalize=False, sample_num=2000,model=model)
    # attack_info = AttackObj.get_adv_data(dataloader=dataloader, method=method, eps=eps)
    # del AttackObj
    # attack = get_attack(method=method, model=model, eps=eps)
    # attack_info = get_adv_data(model=model, attack=attack, dataloader=dataloader, device=device, desc=f'生成{method}对抗样本，eps={eps}')

    return generate_paca_dataloader(attack_info, batch_size)


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

def get_at_dataloader(method='fgsm', model=None, dataloader=None, attackparam = {'eps':1}, device='cuda', batch_size=128, dataset='mnist'):
    torch.manual_seed(seed)
    attack_info = run_get_adv_data(dataset_name=dataset, model = model, dataloader=dataloader, device=device, method=method, attackparam=attackparam)
    # AttackObj = EvasionAttacker(dataset="mnist", device=device, datanormalize=False, sample_num=2000,model=model)
    # attack_info = AttackObj.get_adv_data(dataloader=dataloader, method=method, eps=eps)
    # del AttackObj
    # attack = get_attack(method=method, model=model, eps=eps)
    # attack_info = get_adv_data(model=model, attack=attack, dataloader=dataloader, device=device, desc=f'生成{method}对抗样本，eps={eps}')

    return generate_at_dataloader(attack_info, batch_size)


def generate_cafd_dataloader(attack_info, batch_size):
    dataset = CafdAdversarialDataset(clean_x=attack_info[1], adv_x=attack_info[0], label=attack_info[2])
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    adv_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    adv_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return adv_train_loader, adv_test_loader

def get_cafd_adv_dataloader(method='fgsm', model=None, dataloader=None, attackparam = {'eps':1}, device='cuda', batch_size=128, dataset='mnist'):
    torch.manual_seed(seed)
    attack_info = run_get_adv_data(dataset_name=dataset, model = model, dataloader=dataloader, device=device, method=method, attackparam=attackparam)
    # AttackObj = EvasionAttacker(dataset="mnist", device=device, datanormalize=False, sample_num=2000,model=model)
    # attack_info = AttackObj.get_adv_data(dataloader=dataloader, method=method, eps=eps)
    # del AttackObj
    # attack = get_attack(method=method, model=model, eps=eps)
    # attack_info = get_adv_data(model=model, attack=attack, dataloader=dataloader, device=device, desc=f'生成{method}对抗样本，eps={eps}')
    return generate_cafd_dataloader(attack_info, batch_size)


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


def attack_model(model, dataloader, method, attackparam = {'eps':1}, device='cuda', dataset='mnist'):
    torch.manual_seed(seed)
    attack_info = run_get_adv_data(dataset_name=dataset, model = model, dataloader=dataloader, device=device, method=method, attackparam=attackparam)
    # AttackObj = EvasionAttacker(dataset="mnist", device=device, datanormalize=False, sample_num=2000,model=model)
    # attack_info = AttackObj.get_adv_data(dataloader=dataloader, method=method, eps=eps)
    # del AttackObj
    # attack_method = get_attack(method=method, model=model, eps=eps)
    # attack_info = get_adv_data(model=model, attack=attack_method, dataloader=dataloader, device=device)
    return attack_info


def get_integrate_dataloader(method='fgsm', model=None, dataloader=None, attackparam = {'eps':1}, device='cuda', batch_size=128, dataset='mnist'):
    torch.manual_seed(seed)
    attack_info = run_get_adv_data(dataset_name=dataset, model = model, dataloader=dataloader, device=device, method=method, attackparam=attackparam)
    # AttackObj = EvasionAttacker(dataset="mnist", device=device, datanormalize=False, sample_num=2000,model=model)
    # attack_info = AttackObj.get_adv_data(dataloader=dataloader, method=method, eps=eps)
    # del AttackObj
    # attack = get_attack(method=method, model=model, eps=eps)
    # attack_info = get_adv_data(model=model, attack=attack, dataloader=dataloader, device=device, desc=f'生成{method}对抗样本，eps={eps}')
    data_info = dict()
    data_info['paca'] = generate_paca_dataloader(attack_info, batch_size)
    data_info['cafd'] = generate_cafd_dataloader(attack_info, batch_size)
    data_info['at'] = generate_at_dataloader(attack_info, batch_size)
    data_info['acc'] = attack_info[4]
    return data_info


if __name__ == '__main__':
    from GroupDefense.models.load_model import load_model
    from GroupDefense.datasets.mnist import mnist_dataloader

    model = load_model()
    device = 'cuda'
    model.to(device)
    train_d, test_d = mnist_dataloader(batch_size=128)
    train, test = get_adv_dataloader(method='fgsm', model=model, dataloader=train_d, eps=1, device=device, batch_size=128)
