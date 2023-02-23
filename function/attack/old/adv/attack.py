#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/06/30, ZJUICSR'

from torchvision.models import vgg16

"""
This is the package's interface class for application.
"""
import copy
import os, datetime, pytz, time, sys
import torch
import torch.nn.functional as F
import json
import os.path as osp
from torchvision.utils import save_image, make_grid
from torch.utils.data import TensorDataset
ROOT = osp.dirname(osp.abspath(__file__))
# from attacks import adv
from sklearn.manifold import TSNE
import random
import numpy as np
from function.attack.config import Conf

_, term_width = os.popen('stty size', 'r').read().split()
TOTAL_BAR_LENGTH = 65.  
last_time = time.time()
begin_time = last_time
term_width = int(term_width)

mehods = [
    "FGSM",
    "BIM",
    "RFGSM",
    "CW",
    "PGD",
    "PGDL2",
    "EOTPGD",
    "MultiAttack",
    "FFGSM",
    "TPGD",
    "MIFGSM",
    "VANILA",
    "GN",
    "PGDDLR",
    "APGD",
    "APGDT",
    "FAB",
    "Square",
    "AutoAttack",
    "OnePixel",
    "DeepFool",
    "SparseFool",
    "DI2FGSM"
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)
        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))
    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

class Attack(object):
    def __init__(self, method, params, model, seed=100):
        """
        :param method: attack method
        :param params: An dict with the original inputs to be attacked
        :param model:A trained classifier
        :param seed:Generate random number seed
        """
        
        with open(osp.join(ROOT, "torchattacks/params.json")) as fp:
            def_params = json.load(fp)
        self.seed = seed
        # Helper.set_seed(seed)
        set_seed(seed)
        self.method = method
        self.device = params["device"]
        self.out = params["out_path"]
        self.cache_path = params["cache_path"]
        self.__build__()
        model = model.to(self.device)

        if not method in mehods:
            raise NotImplementedError("Method {:s} not found!".format(method))
        if method == "MultiAttack":
            atk_conf = Conf()
            attacks = []
            if "MultiAttack" in params[method]["attacks"]:
                 raise NotImplementedError("Method {:s} cannot be an attack list member!".format(method))
            for temp in params[method]["attacks"]:
                if not temp in mehods:
                    raise NotImplementedError("Method {:s} not found!".format(temp))
                temp_adv_params = self.__check_params__(atk_conf.get_method_param(temp), def_params[temp].keys())
                attacks.append(eval("adv.{:s}".format(temp))(model, **temp_adv_params))
                del temp_adv_params
            params[method]["attacks"]=copy.deepcopy(attacks)
            del attacks
        
        self.params = params
        # init for attack
        self.adv_params = self.__check_params__(params[method], def_params[method].keys())

        self.attack = eval("adv.{:s}".format(method))(model, **self.adv_params)
        self.attack.set_bounds(self.params["dataset"]["bounds"])
        self.attack.steps = int(self.attack.steps)


    def __check_params__(self, params, keys):
        """
        Filter useless kwargs.
        Args:
            params: dict
            keys: list

        Returns:
            params: dict
        """
        _params = {}
        for k, v in params.items():
            if k in keys:
                _params[k] = v
        return _params

    def __preview__(self, adv_x, real_x=None, size=36):
        """
        Preview generated adversarial examples.
        Args:
            x: Tensor

        Returns:
            plt image
        """
        mean = self.params["dataset"]["mean"]
        std = self.params["dataset"]["std"]

        adv_x = copy.deepcopy(adv_x.to(self.device)).cpu()
        real_x = copy.deepcopy(real_x.to(self.device)).cpu()
        # show perturb pattern
        if real_x is not None:
            for idx in range(1, int(size/3)+1):
                adv_x[idx * 3 - 3] = real_x[idx * 3 - 1]
                adv_x[idx * 3 - 2] = adv_x[idx * 3 - 1] - real_x[idx * 3 - 1]

        for i, (m, s) in enumerate(zip(mean, std)):
            adv_x[:, i, :, :] = adv_x[:, i, :, :] * s + m

        path = osp.join(self.out, "adv_attack_{:s}.jpg".format(self.method))
        print("-> [Attack_{:s}] save preview file:{:s}".format(self.method, path))
        img = make_grid(adv_x[:size], nrow=6)
        save_image(img, fp=path)
        fpath = self.out.split("/")[-1] + "/adv_attack_{:s}.jpg".format(self.method)
        return fpath

    def __preview_single__(self, adv_x, real_x, num=36):
        mean = self.params["dataset"]["mean"]
        std = self.params["dataset"]["std"]

        x = copy.deepcopy(real_x).cpu()[:num]
        try:
            adv_x = copy.deepcopy(adv_x).cpu()[:num]
        except RuntimeError:
            adv_x_1 = adv_x.detach()
            adv_x = copy.deepcopy(adv_x_1).cpu()[:num]

        if type(mean) == type(float(1)):
            x[:, 0, :, :] = x[:, 0, :, :] * std + mean
            adv_x[:, 0, :, :] = adv_x[:, 0, :, :] * std + mean
        else:
            for i, (m, s) in enumerate(zip(mean, std)):
                x[:, i, :, :] = x[:, i, :, :] * s + m
                adv_x[:, i, :, :] = adv_x[:, i, :, :] * s + m

        paths = {}
        preview_path = osp.join(self.out, "preview")
        if not osp.exists(preview_path):
            os.makedirs(preview_path)
        x_len = len(x) if len(x) < 8 else 8
        
        for idx in range(x_len):
            path_real = osp.join(preview_path, "adv_{:s}_real_{:d}.jpg".format(str(self.method), int(idx)))
            path_fake = osp.join(preview_path, "adv_{:s}_fake_{:d}.jpg".format(str(self.method), int(idx)))
            path_noise = osp.join(preview_path, "adv_{:s}_noise_{:d}.jpg".format(str(self.method), int(idx)))

            save_image(x[idx], path_real)
            save_image(adv_x[idx], path_fake)
            save_image(adv_x[idx]-x[idx], path_noise)
            paths[idx] = [path_real.split("/")[-1], path_fake.split("/")[-1], path_noise.split("/")[-1]]
        return paths


    def __build__(self):
        """
        Build path for preview
        Returns:
            success: list, successfully writen path
        """
        paths = [self.out, self.cache_path]
        success = []
        for path in paths:
            if not osp.exists(path):
                print("-> makedirs: {:s}".format(path))
                try:
                    os.makedirs(path)
                    success.append(paths)
                except PermissionError as e:
                    print("-> writing file {:s} error!".format(path))
        return success

    def __call__(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        return self.attack(x, y)

    def preview(self, x, y):
        adv_x = self.__call__(x, y).to(self.device)
        return self.__preview__(adv_x=adv_x, real_x=x)


    def get_adv_loader(self, data_loader, eps=None, cache=False):
        """
        get adv_loader with different eps, load from cache
        :param data_loader:
        :param eps:
        :return: adv_loader
        """
        # copy & backup
        _eps = float(self.attack.eps)
        if eps is not None:
            self.attack.eps = eps
        path = osp.join(self.cache_path,
                        "adv_{:s}_{:s}_{:s}_{:04d}_{:.5f}.pt".format(
                            self.method, self.params["model"]["name"],
                            self.params["dataset"]["name"], int(self.attack.steps), float(self.attack.eps))
                        )

        # try to load from cache
        if cache and osp.exists(path):
            print("-> [Attack_{:s}] generate adv_loader for eps:{:.3f}".format(self.method, eps))
            adv_dst = torch.load(path)
            adv_dst = TensorDataset(adv_dst["x"].cpu(), adv_dst["y"].long().cpu())
        else:
            print(eps)
            print("-> [Attack_{:s}] generate adv_loader for eps:{:.3f}".format(self.method, eps))
            tmp_x, tmp_y = [], []
            for step, (x, y) in enumerate(data_loader):
                # if step==49:
                #     print(x[0])
                #     print("*"*20)
                #     x = self.__call__(x, y).detach().cpu()
                #     print("*"*20)
                #     print(x[0])
                x = self.__call__(x, y).detach().cpu()
                y = y.cpu()
                tmp_x.append(x)
                tmp_y.append(y)
                if step==49:
                    print("*******************")
                    print(y)
                    print("*******************")
                
                progress_bar(
                    step,
                    len(data_loader),
                    "generate with diff eps... eps={:.3f}".format(eps)
                )

            tmp_x = torch.cat(tmp_x)
            tmp_y = torch.cat(tmp_y)
            tmp_dst = {
                "x": tmp_x,
                "y": tmp_y,
                "method": self.method
            }
            torch.save(tmp_dst, path)
            adv_dst = TensorDataset(tmp_x, tmp_y.long())

        adv_loader = torch.utils.data.DataLoader(
            adv_dst,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=2
        )

        # revert eps
        

        self.attack.eps = _eps
        return adv_loader

    def eval_with_eps(self, data_loader, eps=[0.00001, 0.01], steps=15):
        """
        测试随着eps波动，攻击成功率ASR变化，输出配合echart.js: https://echarts.apache.org/examples/en/editor.html?c=line-stack
        attack vars with eps
        :param data_loader:
        :param eps:
        :param steps:
        :return:
        """
        eps_result = {
            "var_asr": [],
            "var_loss": [],
            "var_eps": [],
        }
        step_size = float((eps[1] - eps[0]) / steps)
        self.attack.model.eval()
        for step in range(steps):
            step_eps = eps[0] + step_size * step
            step_correct, step_asr, step_loss = 0, 0.0, 0.0
            print("step_eps:",step_eps)
            adv_loader = self.get_adv_loader(data_loader, eps=step_eps, cache=True)
            with torch.no_grad():
                for i, (x, y) in enumerate(adv_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.attack.model(x)
                    step_correct += pred.argmax(dim=1).view_as(y).eq(y).sum().item()
                    step_loss += F.cross_entropy(pred, y).detach()
                step_asr = 100.0 - 100.0 * (step_correct / len(data_loader.dataset))
                step_loss = step_loss.cpu().numpy() / len(data_loader.dataset)
                eps_result["var_asr"].append(step_asr)
                eps_result["var_loss"].append(step_loss)
                eps_result["var_eps"].append(step_eps)
        return eps_result

    def eval_batch(self, data_loader, num=36):
        x, y = iter(data_loader).next()
        x = x.to(self.device)[:num]
        y = y.to(self.device)[:num]
        adv_x = self.attack(x, y)
        with torch.no_grad():
            pred_ben = self.attack.model(x).detach()
            pred_adv = self.attack.model(adv_x).detach()
            prob_ben = (torch.max(F.softmax(pred_ben, dim=1), dim=1)[0]).cpu().numpy()
            prob_adv = (torch.max(F.softmax(pred_adv, dim=1), dim=1)[0]).cpu().numpy()
            img_paths = self.__preview_single__(adv_x, real_x=x, num=num)

            batch_result = {
                "x_ben": x.cpu().numpy(),
                "x_adv": adv_x.cpu().numpy(),
                "y_ben": torch.argmax(pred_ben, dim=1).view(-1).cpu().numpy().tolist(),
                "y_adv": torch.argmax(pred_adv, dim=1).view(-1).cpu().numpy().tolist(),
                "prob_ben": prob_ben.tolist(),
                "prob_adv": prob_adv.tolist(),
                "img_paths": img_paths
            }
            return batch_result

    def eval_first_batch(self, data_loader):
        x, y = iter(data_loader).next()
        x = x.to(self.device)
        _model = self.attack.model.to(self.device)
        with torch.no_grad():
            logits = _model(x)
            prob = F.softmax(logits, dim=1).detach().cpu()
            return prob

    @staticmethod
    def prob_scatter(ben_prob, adv_prob, seed=100):
        _min = int(min(len(ben_prob), len(adv_prob)))
        ben_prob = ben_prob[:_min]
        adv_prob = adv_prob[:_min]
        probs = torch.cat([ben_prob, adv_prob]).cpu().numpy()
        x = TSNE(n_components=2, n_iter=1000, random_state=seed).fit_transform(probs)
        return x

    def run(self, data_loader, eps=[0.00001, 0.1], num=36):
        # get adv_loader
        adv_loader = self.get_adv_loader(data_loader, eps=float(self.attack.eps), cache=False)

        # 执行攻击机理分析时，可注释掉
        # eval mini-batch   
        result_batch = self.eval_batch(data_loader, num=num)
        # eval vars with eps
        result_eps = self.eval_with_eps(data_loader, eps=eps, steps=15)

        result_final = {
            "adv_loader": adv_loader,
        }
        # 执行攻击机理分析时，可注释掉
        print("-> Attack ASR={:s}".format(str(result_eps['var_asr'])))
        result_final.update(result_eps)
        result_final.update(result_batch)
        return result_final