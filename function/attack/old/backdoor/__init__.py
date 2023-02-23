#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/30, ZJUICSR'

"""
Note: Wrapper for different backdoor attacks
"""
import os.path as osp
from halo import Halo
from torchvision.utils import save_image, make_grid
from .badnets.badnets_v2 import Badnets


class WrapperBackdoorAttack:
    def __init__(self, args, method, params, **kwargs):
        self.args = args
        self.method = method
        self.params = params
        self.out_path = params["sys"]["out_path"]
        for k, v in kwargs.items():
            setattr(self, k, v)

    def preview(self, atk, data_loader):
        assert self.atk is not None
        clean_x, clean_y = list(data_loader)[0]
        poison_x, poison_y, plabel = atk.poison_batch(
            clean_x, clean_y, target=atk.target,
            idxs=list(range(data_loader.batch_size))
        )
        return self.__preview__(poison_x, clean_x)

    def __preview__(self, poison_x, clean_x=None, size=36):
        """
        Preview generated adversarial examples.
        Args:
            x: Tensor

        Returns:
            plt image
        """
        mean = self.params["dataset"]["mean"]
        std = self.params["dataset"]["std"]
        poison_x = poison_x.cpu()
        clean_x = clean_x.cpu()

        # show perturb pattern
        if poison_x is not None:
            for idx in range(1, int(size/2)):
                poison_x[idx * 2 - 2] = clean_x[idx * 2 - 1]

        for i, (m, s) in enumerate(zip(mean, std)):
            poison_x[:, i, :, :] = poison_x[:, i, :, :] * s + m

        path = osp.join(self.out_path, f"backdoor_preview_{self.method}.jpg")
        print(f"-> save preview file:{path}")
        img = make_grid(poison_x[:size], nrow=6)
        save_image(img, format="JPEG", fp=path)
        fpath = self.out_path.split("/")[-1] + f"/adv_attack_{self.method}.jpg"
        return fpath

    def attack_batch(self, model, data_loader, **kwargs):
        if self.method == "BadNets":
            # for badnets attack
            atk = Badnets(model=model, data_loader=data_loader,
                          **self.params[self.method]
                          )
            clean_x, clean_y = list(data_loader)[0]
            poison_x, _, _ = atk.poison_batch(
                clean_x, clean_y, target=atk.target,
                idxs=list(range(data_loader.batch_size))
            )
            self.preview_path = self.preview(atk, data_loader)
            return poison_x

    def attack(self, model, data_loader, train=True):
        if self.method == "BadNets":
            self.atk = Badnets(model=model, data_loader=data_loader,
                               **self.params[self.method])
            poison_loader = self.atk.poison(shuffle=True, train=train)
            return poison_loader


    def __call__(self, model, train_loader, test_loader, **kwargs):
        step_result = {"images": []}
        if self.method == "BadNets":
            p_train_loader = self.attack(model, train_loader)
            p_test_loader = self.attack(model, test_loader, train=False)
            p_batch = self.attack_batch(model, train_loader)
            step_result["images"].append({
                "title": "backdoor_images",
                "path": self.preview_path.split("/")[-2] + "/" + self.preview_path.split("/")[-1],
                "alt": f"Backdoor attack:{self.method} backdoor images preview"
            })
            return p_train_loader, p_test_loader, p_batch, step_result





