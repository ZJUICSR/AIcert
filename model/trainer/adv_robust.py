#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/01, ZJUICSR'


import copy
import numpy as np
import torch
import torch.nn.functional as F
from models.trainer.trainer import Trainer
import random
from IOtool import IOtool

class RobustTrainer(Trainer):
    def __init__(self, **kwargs):
        super(RobustTrainer, self).__init__(**kwargs)

    def robust_train_advloader(self, model, ben_train_loader, adv_train_loader, ben_test_loader, adv_test_loader, epochs=50, epoch_fn=None, rate=0.4, **kwargs):
        """
        用于用户上传的对抗样本做鲁棒训练
        :param model:
        :param ben_train_loader:
        :param adv_train_loader:
        :param ben_test_loader:
        :param adv_test_loader:
        :param epochs:
        :param epoch_fn:
        :param rate:
        :param kwargs:
        :return:
        """
        device = self.device
        model = model.cpu()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr * 0.3, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_acc = 0.0
        best_loss = 0.0
        best_epoch = 0
        train_list = []
        test_list = []
        print(f"-> 开始鲁棒训练，服务运行在显卡:{self.device}")
        for epoch in range(1, epochs + 1):
            model.train()
            model = model.to(device)

            num_step = len(ben_train_loader)
            total, sum_correct, sum_loss = 0, 0, 0.0

            iter_adv_train_loader = iter(adv_train_loader)
            for step, (x, y) in enumerate(ben_train_loader):
                try:
                    adv_x, adv_y = iter_adv_train_loader.next()
                except Exception as e:
                    iter_adv_train_loader = iter(adv_train_loader)
                    adv_x, adv_y = iter_adv_train_loader.next()
                size = int(rate * len(x))
                idx = np.random.choice(len(x), size=size, replace=False)
                x[idx] = adv_x[idx].cpu()
                y[idx] = adv_y[idx].cpu()

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

                total += y.size(0)
                sum_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                sum_correct += pred.eq(y.view_as(pred)).sum().item()
                info = "[Train] 针对用户上传的对抗样本做鲁棒训练， Loss: {:.6f} Acc:{:.3f}%".format(
                    sum_loss / total,
                    100.0 * (sum_correct / total)
                )
                IOtool.progress_bar(step, num_step, info)
            train_acc, train_loss = 100.0 * (sum_correct / total), sum_loss / total
            self.train_res["epoch"].append(epoch)
            self.train_res["acc"].append(train_acc)
            self.train_res["loss"].append(train_loss)
            test_acc, test_loss = self.test(model, ben_test_loader)
            adv_test_acc, adv_test_loss = self.test(model, adv_test_loader)

            if best_acc < test_acc:
                best_acc = test_acc
                best_loss = test_loss
                best_epoch = epoch

            train_list.append(train_acc)
            test_list.append(test_acc)
            epoch_result = {
                "epoch": epoch,
                "best_acc": best_acc,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "train": [train_acc, train_loss],
                "test": [test_acc, test_loss],
                "train_list": train_list,
                "test_list": test_list
            }
            if epoch_fn is not None:
                try:
                    eval("epoch_fn(model, epoch_result, **kwargs)")
                except Exception as e:
                    print("-> Trainer.train() callback error: {:s}!".format(str(e)))
            scheduler.step()
            print()
        model = model.cpu()
        return model


    def robust_train(self, model, train_loader, test_loader, adv_loader, epochs=40, atk=None, epoch_fn=None, rate=0.25, **kwargs):
        """
        用于系统内定的对抗算法做鲁棒训练
        :param model:
        :param train_loader:
        :param test_loader:
        :param adv_loader:
        :param epochs:
        :param atk:
        :param epoch_fn:
        :param rate:
        :param kwargs:
        :return:
        """
        assert "atk_method" in kwargs.keys()
        assert "def_method" in kwargs.keys()
        device = self.device
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr * 0.3, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_acc = 0.0
        best_loss = 0.0
        best_epoch = 0
        train_list = []
        test_list = []
        print(f"-> 开始鲁棒训练，服务运行在显卡:{self.device}")
        _eps = copy.deepcopy(atk.attack.eps)
        for epoch in range(1, epochs + 1):
            print("-> For epoch:{:d} adv training on device: {:s}".format(epoch, str(self.device)))
            model.train()
            model = model.to(device)
            num_step = len(train_loader)
            total, sum_correct, sum_loss = 0, 0, 0.0
            for step, (x, y) in enumerate(train_loader):
                if atk is not None:
                    size = int(rate * len(x))
                    idx = np.random.choice(len(x), size=size, replace=False)
                    atk.attack.eps = _eps * (random.randint(80, 180) * 0.01)
                    x, y = x.to(device), y.to(device)
                    x[idx] = atk.attack(copy.deepcopy(x[idx]), copy.deepcopy(y[idx]))

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

                total += y.size(0)
                sum_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                sum_correct += pred.eq(y.view_as(pred)).sum().item()
                info = "[Train] Attack:{:s}_{:.4f} Defense:{:s} Loss: {:.6f} Acc:{:.3f}%".format(
                    kwargs["atk_method"],
                    atk.attack.eps,
                    kwargs["def_method"],
                    sum_loss / total,
                    100.0 * (sum_correct / total)
                )
                IOtool.progress_bar(step, num_step, info)
            atk.attack.eps = _eps
            train_acc, train_loss = 100.0 * (sum_correct / total), sum_loss / total
            self.train_res["epoch"].append(epoch)
            self.train_res["acc"].append(train_acc)
            self.train_res["loss"].append(train_loss)
            test_acc, test_loss = self.test(model, test_loader)
            adv_test_acc, adv_test_loss = self.test(model, adv_loader)

            if best_acc < test_acc:
                best_acc = test_acc
                best_loss = test_loss
                best_epoch = epoch

            train_list.append(train_acc)
            test_list.append(test_acc)
            epoch_result = {
                "epoch": epoch,
                "best_acc": best_acc,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "train": [train_acc, train_loss],
                "test": [test_acc, test_loss],
                "train_list": train_list,
                "test_list": test_list
            }
            if epoch_fn is not None:
                try:
                    eval("epoch_fn(model, epoch_result, **kwargs)")
                except Exception as e:
                    print("-> Trainer.train() callback error: {:s}!".format(str(e)))
            scheduler.step()
            print()
        model = model.cpu()
        return model