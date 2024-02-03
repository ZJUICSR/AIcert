#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/01, ZJUICSR'

'''
Base trainer object, only support for vision training and evaluation
'''
# TODO: extend to NLP model, extend to tensorflow

import torch
import torch.nn.functional as F
from IOtool import IOtool
# from argp.utils.helper import Helper
from model.trainer.base import BaseTrainer
best_acc = 0.0

class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        # set default attribution
        self.train_res = {
            "epoch": [],
            "acc": [],
            "loss": []
        }
        self.test_res = {
            "acc": [],
            "loss": []
        }
        self.__check_params__()

    @staticmethod
    def __config__():
        """
        Returns: dict
            Default config params
        """
        params = {
            "lr": 0.1,
            "optim": "SGD",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "device": IOtool.get_device()
        }
        return params

    def update_config(self, arch, task):
        params = {
            "resnet_mnist": {
                "lr": 0.005,
                "optim": "SGD",
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "device": IOtool.get_device()
            },
            "vgg_mnist": {
                "lr": 0.01,
                "optim": "SGD",
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "device": IOtool.get_device()
            },
            "resnet_cifar": {
                "lr": 0.05,
                "optim": "SGD",
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "device": IOtool.get_device()
            },
            "vgg_cifar": {
                "lr": 0.1,
                "optim": "SGD",
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "device": IOtool.get_device()
            }
        }

        _params = {}
        if "resnet" in arch.lower():
            if "cifar" in task.lower():
                _params = params["resnet_cifar"]
            elif "mnist" in task.lower():
                _params = params["resnet_mnist"]

        elif "vgg" in arch.lower():
            if "cifar" in task.lower():
                _params = params["vgg_cifar"]
            elif "mnist" in task.lower():
                _params = params["vgg_mnist"]

        for k, v in _params.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        print("-> update trainer config[{:s}.{:s}]: {:s}".format(arch.lower(), task.lower(), str(_params)))


    @staticmethod
    def callback_train(model, epoch_result, **kwargs):
        """
        callback example for train function
        :param model:
        :param train_result:
        :param kwargs:
        :return:
        """
        if "results" in kwargs.keys():
            kwargs["results"] = epoch_result


    def pretrain(self, model, train_loader, test_loader, epochs=350):
        best_acc = 0.0
        self.lr = 0.1
        def _train(model, optimizer, train_loader, epoch, device, backends=False):
            print('\nEpoch: %d' % epoch)
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if backends:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                else:
                    inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                IOtool.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            return model

        def _test(model, test_loader, device, backends=False):
            global best_acc
            test_loss = 0
            correct = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if backends:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    else:
                        inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    IOtool.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% Best_Acc: %.3f%% (%d/%d)'
                                        % (
                                        test_loss / (batch_idx + 1), 100. * correct / total, best_acc, correct, total))
            # Save checkpoint.
            acc = 100. * correct / total
            return acc

        start_epoch = 0
        split_step = [150, 250, 350]
        device = self.device
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(start_epoch, start_epoch + split_step[0]):
            print("-> For epoch:{:d} in device: {:s}".format(epoch, str(self.device)))
            _train(model, optimizer, train_loader, epoch, device=device)
            acc = _test(model, test_loader, device=device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr / 10.0, momentum=0.9, weight_decay=self.weight_decay)
        for epoch in range(start_epoch + split_step[0], start_epoch + split_step[1]):
            print("-> For epoch:{:d} in device: {:s}".format(epoch, str(self.device)))
            _train(model, optimizer, train_loader, epoch, device=device)
            acc = _test(model, test_loader, device=device)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr / 100.0, momentum=0.9, weight_decay=self.weight_decay)
        for epoch in range(start_epoch + split_step[1], start_epoch + split_step[2]):
            print("-> For epoch:{:d} in device: {:s}".format(epoch, str(self.device)))
            _train(model, optimizer, train_loader, epoch, device=device)
            acc = _test(model, test_loader, device=device)
        return model


    def train(self, model, train_loader, test_loader, epochs=100, pre_fn=None, epoch_fn=None,logging=None, **kwargs):
        device = self.device
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        best_loss = 0.0
        best_epoch = 0
        train_list = []
        test_list = []
        for epoch in range(1, epochs + 1):
            model.train()
            model = model.to(device)
            num_step = len(train_loader)
            
            total, sum_correct, sum_loss = 0, 0, 0.0
            print("-> For epoch:{:d} on device: {:s}".format(epoch, str(self.device)))
            for step, (x, y) in enumerate(train_loader):
                if pre_fn is not None:
                    x, y = eval("pre_fn(model, x, y, **kwargs)")
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
                info = "[Train] Loss: {:.6f} Acc:{:.3f}%".format(
                    sum_loss / total,
                    100.0 * (sum_correct / total)
                )
                IOtool.progress_bar(step, num_step, info)
            train_acc, train_loss = 100.0 * (sum_correct / total), sum_loss / total
            self.train_res["epoch"].append(epoch)
            self.train_res["acc"].append(train_acc)
            self.train_res["loss"].append(train_loss)
            test_acc, test_loss = self.test(model, test_loader)

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
                    eval("epoch_fn(model, epoch_result, logging=logging,**kwargs)")
                except Exception as e:
                    print("-> Trainer.train() callback error: {:s}!".format(str(e)))
            scheduler.step()
            print()
        return model

    def test(self, model, test_loader, device=None, **kwargs):
        device = self.device if device is None else device
        total, sum_correct, sum_loss = 0, 0, 0.0
        model.to(device)
        num_step = len(test_loader)
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = F.cross_entropy(output, y)
                total += y.size(0)
                sum_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                sum_correct += pred.eq(y.view_as(pred)).sum().item()
                info = "[Test] Loss: {:.6f} Acc:{:.3f}%".format(
                    sum_loss / total,
                    100.0 * sum_correct / total
                )
                IOtool.progress_bar(step, num_step, info)
        test_loss = sum_loss / (1.0 * total)
        test_acc = 100.0 * sum_correct / total
        self.test_res["acc"].append(100.0 * (sum_correct / total))
        self.test_res["loss"].append(sum_loss / total)
        del x
        del y
        model = model.cpu()
        return round(float(test_acc), 3), round(float(test_loss), 5)

    def test_batch(self, model, test_loader, device=None, **kwargs):
        device = self.device if device is None else device
        model.to(device)
        with torch.no_grad():
            x, y = iter(test_loader).next()
            x = x.to(device)
            output = model(x).detach().cpu()
            loss = F.cross_entropy(output, y).detach().cpu()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            acc = 100.0 * float(correct / len(x))
            return output, round(float(acc), 3), round(float(loss), 5)














