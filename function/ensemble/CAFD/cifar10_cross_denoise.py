# -*- coding: utf-8 -*-
from glob import glob
import os

import json
from os.path import join, dirname

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.denoiser import Denoiser
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from networks import Wide_ResNet
from cifar_data import get_cifar_fuse_dataloader


RESULTS_FOLDER = join(dirname(__file__), 'result')
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)


RESULTS_FOLDER = join(dirname(__file__), 'result')
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)


class ModelScore(object):
    def __init__(self, model: nn.Module, dataloader: DataLoader, device='cuda', denoise=None):
        self.device = device
        self.model = model
        self.model.eval()
        self.denoise = denoise
        self.y_label, self.y_output, self.y_pre, self.adv_y_output, self.adv_y_pre = self.test_model(dataloader=dataloader, device=device)

    def cal_roc(self, adv=True):
        cls = len(self.y_output[0])
        for i in range(cls):
            fpr, tpr, _ = roc_curve(self.y_label, [self.y_output[j][i] for j in range(len(self.y_output))], pos_label=i)
            roc_auc = auc(fpr, tpr)
            print('ok')
            return fpr, tpr, roc_auc        # 只进行一次roc

    def cal_acc(self, adv=True):
        if adv:
            return accuracy_score(self.y_label, self.adv_y_pre)
        return accuracy_score(self.y_label, self.y_pre)

    def cal_precision(self, adv=True):
        if adv:
            return precision_score(self.y_label, self.adv_y_pre, average='macro')
        return precision_score(self.y_label, self.y_pre, average='macro')

    def cal_recall(self, adv=True):
        if adv:
            return recall_score(self.y_label, self.adv_y_pre, average='macro')
        return recall_score(self.y_label, self.y_pre, average='macro')

    def cal_f1(self, adv=True):
        if adv:
            return f1_score(self.y_label, self.adv_y_pre, average='macro')
        return f1_score(self.y_label, self.y_pre, average='macro')

    def cal_auc(self, adv=True):
        if adv:
            return roc_auc_score(self.y_label, self.adv_y_output, average='macro', multi_class='ovo')
        return roc_auc_score(self.y_label, self.y_output, average='macro', multi_class='ovo')

    # @torch.no_grad()
    def test_model(self, dataloader, device):
        save_target, x_save_output, x_save_pre, adv_save_output, adv_save_pre = list(), list(), list(), list(), list()
        m = nn.Softmax(dim=1)

        for x, x_adv, label in tqdm(dataloader, ncols=100, desc="Evaluate model", leave=False, disable=False):
            x, x_adv, label = x.to(device), x_adv.to(device), label.to(device)

            x = self.denoise(x)
            x_adv = self.denoise(x_adv)

            x_output = self.model(x)
            adv_output = self.model(x_adv)

            # 保存待计算的数据
            save_target.extend(list(label.detach().cpu().numpy()))

            x_save_output.extend(list((m(x_output)).detach().cpu().numpy()))
            x_save_pre.extend(list(torch.max(m(x_output), dim=1).indices.detach().cpu().numpy()))

            adv_save_output.extend(list((m(adv_output)).detach().cpu().numpy()))
            adv_save_pre.extend(list(torch.max(m(adv_output), dim=1).indices.detach().cpu().numpy()))

        return save_target, x_save_output, x_save_pre, adv_save_output, adv_save_pre


class DENOISE(torch.nn.Module):
    def __init__(self, denoise, device='cuda'):
        super(DENOISE, self).__init__()
        self.device = device
        self.denoise = denoise.to(device)
        self.denoise.eval()

    def forward(self, x):
        noise = self.denoise.forward(x)
        x_smooth = x + noise
        return x_smooth


def cross_cafd(dataset_name='cifar10', eps=2, device='cuda'):
    file_name = glob(os.path.join(dirname(__file__),
                                  'checkpoint_denoise',
                                  f'CAFD',
                                  f'cifar10_*_{eps}_cafd.pth.tar'))
    file_name += glob(os.path.join(dirname(__file__),
                                  'checkpoint_denoise',
                                  f'CAFD',
                                  f'cifar10_cw_*_cafd.pth.tar'))
    file_name += glob(os.path.join(dirname(__file__),
                                  'checkpoint_denoise',
                                  f'CAFD',
                                  f'cifar10_deepfool_*_cafd.pth.tar'))
    # print(f"file_name={file_name}, dir={os.path.join(dirname(__file__), 'checkpoint_denoise', f'CAFD' f'cifar10_cw_*_cafd.pth.tar')}")
    results = dict()
    net = Wide_ResNet(depth=16, widen_factor=10, dropout_rate=0.3, num_classes=10)
    checkpoints = torch.load("/opt/data/user/gss/code/cyber_adv/Comdefend/checkpoints/cifar10/wide-resnet-16x10.pt")
    net.load_state_dict(checkpoints['model'])
    net.cuda()
    net.eval()
    all_methods = [f.split(os.sep)[-1].split("_")[1] for f in file_name]
    result_file = os.path.join(RESULTS_FOLDER, f'{dataset_name}_cafd_corss_denoise_result.json')
    for pt_file in tqdm(file_name, desc='Calc Performance', ncols=100):
        denoiser = Denoiser(x_h=32, x_w=32, channel=3)
        denoiser.cuda()
        denoiser = torch.nn.DataParallel(denoiser, device_ids=range(torch.cuda.device_count()))

        checkpoint = torch.load(pt_file)
        denoiser.load_state_dict(checkpoint)
        denoiser.cuda()
        denoiser.eval()

        denoise = DENOISE(denoise=denoiser, device=device)

        file_name = pt_file.split(os.sep)[-1].split("_")
        attack_method = file_name[1]
        if attack_method in ['fgsm']:
            continue
        for method in all_methods:
            data_eps = eps if method not in ['cw', 'deepfool'] else 2
            trainloader, testloader = get_cifar_fuse_dataloader(method=method,
                                                              eps=data_eps,
                                                              batch_size=100,
                                                              train_total=20000,
                                                                test_total=5000)
            cal_score = ModelScore(model=net,
                                   dataloader=trainloader,
                                   device=device,
                                   denoise=denoise)

            denoise_results = {'adv': {'acc': cal_score.cal_acc(),
                                        'pre': cal_score.cal_precision(),
                                        'recall': cal_score.cal_recall(),
                                        'f1': cal_score.cal_f1(),
                                        'auc': cal_score.cal_auc()},
                                'ori': {'acc': cal_score.cal_acc(adv=False),
                                        'pre': cal_score.cal_precision(adv=False),
                                        'recall': cal_score.cal_recall(adv=False),
                                        'f1': cal_score.cal_f1(adv=False),
                                        'auc': cal_score.cal_auc(adv=False)}
                              }

            cal_score = ModelScore(model=net,
                                   dataloader=testloader,
                                   device=device,
                                   denoise=denoise)

            test_results = {'adv': {'acc': cal_score.cal_acc(),
                                        'pre': cal_score.cal_precision(),
                                        'recall': cal_score.cal_recall(),
                                        'f1': cal_score.cal_f1(),
                                        'auc': cal_score.cal_auc()},
                                'ori': {'acc': cal_score.cal_acc(adv=False),
                                        'pre': cal_score.cal_precision(adv=False),
                                        'recall': cal_score.cal_recall(adv=False),
                                        'f1': cal_score.cal_f1(adv=False),
                                        'auc': cal_score.cal_auc(adv=False)}
                              }
            if f'{eps}' not in results:
                results[f'{eps}'] = dict()
            if attack_method not in results[f'{eps}']:
                results[f'{eps}'][attack_method] = dict()
            results[f'{eps}'][attack_method][method] = {'train': denoise_results, 'test': test_results}

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)


if __name__ == '__main__':
    # cross_cafd(eps=2)
    cross_cafd(eps=8)

