# -*- coding: utf-8 -*-
# @Time    : 2022/5/10 16:35
# @File    : roc_auc.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


class ModelScore(object):
    def __init__(self, model: nn.Module, dataloader: DataLoader, device='cuda',
                 binary=True, smooth=False, smooth_value=16,
                 need_grad=True, victim_model=None):
        self.binary = binary
        self.need_grad = need_grad
        self.victim_model = victim_model
        self.victim_model.to(device)
        self.victim_model.eval()
        self.device = device
        self.y_output, self.y_label, self.y_pre = self.test_model(model=model,
                                                                  dataloader=dataloader,
                                                                  device=device,
                                                                  smooth=smooth,
                                                                  smooth_value=smooth_value)

    def cal_roc(self):
        cls = len(self.y_output[0])
        for i in range(cls):
            fpr, tpr, _ = roc_curve(self.y_label, [self.y_output[j][i] for j in range(len(self.y_output))], pos_label=i)
            roc_auc = auc(fpr, tpr)
            print('ok')
            return fpr, tpr, roc_auc        # 只进行一次roc

    def cal_acc(self):
        return accuracy_score(self.y_label, self.y_pre)

    def cal_precision(self):
        return precision_score(self.y_label, self.y_pre, average='macro' if not self.binary else 'binary')

    def cal_recall(self):
        return recall_score(self.y_label, self.y_pre, average='macro' if not self.binary else 'binary')

    def cal_f1(self):
        return f1_score(self.y_label, self.y_pre, average='macro' if not self.binary else 'binary')

    def cal_auc(self):
        if self.binary:
            try:
                return roc_auc_score(self.y_label, np.array(self.y_output)[:, 1])
            except Exception as e:
                print(f'error as {e}')
                # print(f'y_label={self.y_label}')
                # print(f'y_output={self.y_output}')
        else:
            return roc_auc_score(self.y_label, self.y_output, average='micro')

    def calc_pred_confident(self):
        # print(f'self.y_output={self.y_output}')
        pred_data, _ = torch.sort(torch.tensor(self.y_output), descending=True)
        return (pred_data[:, 0] / pred_data[:, 1]).tolist()

    # @torch.no_grad()
    def test_model(self, model, dataloader, device, smooth=False, smooth_value=16):
        save_target, save_output,  save_pre = list(), list(), list()
        m = nn.Softmax(dim=1)
        model.to(device)
        # model.eval()
        criterion = nn.CrossEntropyLoss()

        smooth_fun = lambda data, smooth_value: (((data * 255).int() / smooth_value).int() * smooth_value) / 255
        for images, target in tqdm(dataloader, ncols=100, desc="Evaluate model", leave=False, disable=False):
            if smooth:
                images = smooth_fun(images, smooth_value)
            images = images.to(device)
            # target = target.to(device)
            # compute output
            # with torch.cuda.amp.autocast():
            if self.need_grad:
                images = torch.autograd.Variable(images, requires_grad=True)
                y_logit = self.victim_model(images)
                _, pred_class = torch.max(y_logit, 1)
                loss = criterion(y_logit, pred_class)
                gradient = torch.autograd.grad(loss, images)[0]
                gradient = torch.abs(gradient).detach().to(device)
                with torch.no_grad():
                    output = model(images, gradient)
            else:
                output = model(images)

            # 保存待计算的数据
            save_output.extend(list((m(output)).detach().cpu().numpy()))
            save_target.extend(list(target.detach().cpu().numpy()))
            save_pre.extend(list(torch.max(m(output), dim=1).indices.detach().cpu().numpy()))
        # print(f'*****acc = {np.array(train_acc).mean():.6f}')
        # print(save_output)
        return save_output, save_target, save_pre


if __name__ == '__main__':
    pass

