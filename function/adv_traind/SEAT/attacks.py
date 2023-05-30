from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from utils import torch_accuracy


class AttackerPolymer:

    def __init__(self, epsilon, num_steps, step_size, num_classes, device):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_classes = num_classes
        self.device = device

        self.attacker_name = ['NAT', 'PGD_20', 'PGD_100', 'MIM', 'CW', 'AA']

    def run_all(self, model, img, gt, return_acc=True):
        adv_acc_dict = {}

        adv_acc_dict['NAT'] = self.NAT(model, img, gt, return_acc=return_acc)
        adv_acc_dict['PGD_20'] = self.PGD(model, img, gt, num_steps=20, category='Madry', return_acc=return_acc)
        adv_acc_dict['PGD_100'] = self.PGD(model, img, gt, num_steps=100, category='Madry', return_acc=return_acc)
        adv_acc_dict['MIM'] = self.MIM(model, img, gt, return_acc=return_acc)
        adv_acc_dict['CW'] = self.CW(model, img, gt, return_acc=return_acc)
        adv_acc_dict['APGD_ce'] = self.AA(model, img, gt, attacks_to_run=['apgd-ce'], return_acc=return_acc)
        adv_acc_dict['APGD_dlr'] = self.AA(model, img, gt, attacks_to_run=['apgd-dlr'], return_acc=return_acc)
        adv_acc_dict['APGD_t'] = self.AA(model, img, gt, attacks_to_run=['apgd-t'], return_acc=return_acc)
        adv_acc_dict['FAB_t'] = self.AA(model, img, gt, attacks_to_run=['fab-t'], return_acc=return_acc)
        adv_acc_dict['Square'] = self.AA(model, img, gt, attacks_to_run=['square'], return_acc=return_acc)
        adv_acc_dict['AA'] = self.AA(model, img, gt, return_acc=return_acc)

        return adv_acc_dict

    def run_specified(self, name, model, img, gt, step_count=None, category='Madry', return_acc=False):
        name = name.upper()
        if 'PGD' in name:
            num_steps = int(name.split('_')[-1])
            return self.PGD(model, img, gt, num_steps=num_steps, category=category, step_count=step_count, return_acc=return_acc)
        elif name == 'MIM':
            return self.MIM(model, img, gt, return_acc=return_acc)
        elif name == 'CW':
            return self.CW(model, img, gt, return_acc=return_acc)
        elif name == 'AA':
            return self.AA(model, img, gt, return_acc=return_acc)
        elif name == 'NAT':
            return self.NAT(model, img, gt, return_acc=return_acc)
        else:
            raise NotImplementedError

    def NAT(self, model, img, gt, return_acc=False):
        model.eval()
        if return_acc:
            pred = model(img)
            acc = torch_accuracy(pred, gt, (1,))
            return acc
        return img

    def PGD(self, model, img, gt, num_steps, category='Madry', rand_init=True, step_count=None, return_acc=False):
        model.eval()
        if category == "trades":
            x_adv = img.detach() + 0.001 * torch.randn(img.shape).to(self.device).detach() if rand_init else img.detach()
            nat_output = model(img)
        elif category == "Madry":
            x_adv = img.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img.shape)).float().to(self.device) if rand_init else img.detach()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise NotImplementedError
        
        for k in range(self.num_steps):
            x_adv.requires_grad_()
            output = model(x_adv)

            model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.CrossEntropyLoss()(output, gt)
                loss_adv.backward()

            if step_count is not None:
                step_count += torch.eq(output.max(1)[1], gt).int()

            eta = self.step_size * x_adv.grad.sign()
            # Update adversarial img
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, img - self.epsilon), img + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(x_adv, requires_grad=False)
        if return_acc:
            adv_pred = model(x_adv)
            adv_acc = torch_accuracy(adv_pred, gt, (1,))
            return adv_acc
        if step_count:
            return x_adv, step_count
        else:
            return x_adv

    def MIM(self, model, img, gt, decay_factor=1.0, return_acc=False):
        model.eval()
        x_adv = img.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img.shape)).float().to(self.device)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        previous_grad = torch.zeros_like(img.data)

        for k in range(self.num_steps):
            x_adv.requires_grad_()
            output = model(x_adv)

            model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.CrossEntropyLoss()(output, gt)
            loss_adv.backward()
            grad = x_adv.grad.data / torch.mean(torch.abs(x_adv.grad.data), [1,2,3], keepdim=True)
            previous_grad = decay_factor * previous_grad + grad
            eta = self.step_size * previous_grad.sign()
            # Update adversarial img
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, img - self.epsilon), img + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(x_adv, requires_grad=False)
        if return_acc:
            adv_pred = model(x_adv)
            adv_acc = torch_accuracy(adv_pred, gt, (1,))
            return adv_acc
        return x_adv


    def CW(self, model, img, gt, margin=50, return_acc=False):
        model.eval()
        x_adv = Variable(img.data, requires_grad=True)
        random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
        x_adv = Variable(x_adv.data + random_noise, requires_grad=True)
        onehot_targets = torch.eye(self.num_classes)[gt].to(self.device)

        for _ in range(self.num_steps):
            opt = optim.SGD([x_adv], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                logits = model(x_adv)

                self_loss = torch.sum(onehot_targets * logits, dim=1)
                other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

                loss = -torch.sum(torch.clamp(self_loss - other_loss + margin, 0))
                loss = loss / onehot_targets.shape[0]

            loss.backward()
            eta = self.step_size * x_adv.grad.data.sign()
            x_adv = Variable(x_adv.data + eta, requires_grad=True)
            eta = torch.clamp(x_adv.data - img.data, -self.epsilon, self.epsilon)
            x_adv = Variable(img.data + eta, requires_grad=True)
            x_adv = Variable(torch.clamp(x_adv, 0, 1.0), requires_grad=True)
        if return_acc:
            adv_pred = model(x_adv)
            adv_acc = torch_accuracy(adv_pred, gt, (1,))
            return adv_acc
        return x_adv


    def AA(self, model, img, gt, attacks_to_run=None, return_acc=False):
        try:
            from autoattack import AutoAttack
        except:
            os.system('pip3 install git+https://github.com/fra31/auto-attack')
            from autoattack import AutoAttack
        
        adversary = AutoAttack(model, norm='Linf', eps=self.epsilon, version='standard', verbose=False)
        if attacks_to_run:
            adversary.attacks_to_run = attacks_to_run   
            x_adv = adversary.run_standard_evaluation_individual(img, gt, bs=len(img))[attacks_to_run[0]]
        else:
            x_adv = adversary.run_standard_evaluation(img, gt, bs=len(img))
        if return_acc:
            adv_pred = model(x_adv)
            adv_acc = torch_accuracy(adv_pred, gt, (1,))
            return adv_acc
        return x_adv