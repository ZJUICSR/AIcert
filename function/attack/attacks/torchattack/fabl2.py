from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import Optional
import torch
import torch.nn.functional as F
# zero_gradients deprecated in torch >= 1.9.
# zero_gradients is re-defined in the bottom of the code.
# from torch.autograd.gradcheck import zero_gradients
from collections import abc as container_abcs
import numpy as np
from function.attack.attacks.attack import EvasionAttack


class FABL2(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "eps"
    ]
    r"""
    Fast Adaptive Boundary Attack in the paper 'Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack'
    [https://arxiv.org/abs/1907.02044]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2, L1

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2', 'L1'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        alpha_max (float): alpha_max. (Default: 0.1)
        eta (float): overshooting. (Default: 1.05)
        beta (float): backward step. (Default: 0.9)

        seed (int): random seed for the starting point. (Default: 0)
        targeted (bool): targeted attack for every wrong classes. (Default: False)
        n_classes (int): number of classes. (Default: 10)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = FAB(model, steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, loss_fn=None, seed=0, targeted=False, n_classes=10)
        >>> adv_images = attack.generate(x, y)

    """

    def __init__(self, classifier, eps=None, steps=10, n_restarts=1,
                 alpha_max=0.1, eta=1.05, beta=0.9, seed=0,
                 multi_targeted=False, n_classes=10):
       
        self.norm = 'L2'
        self.n_restarts = n_restarts
        self.eps = eps if eps is not None else 1.0
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.steps = steps
        self.seed = seed
        self.target_class = None
        self.multi_targeted = multi_targeted
        self.n_target_classes = n_classes - 1
        self.classifier = classifier
        self.device = classifier.device

    def generate(self, x : np.ndarray, y : Optional[np.ndarray] = None):
        r"""
        Overridden.
        """
        if y is None:
            y = self.classifier.predict(x)
        
        if len(y.shape) == 2:
            y = self.classifier.reduce_labels(y) 
            
        x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y).to(self.device)

        adv_x = self.perturb(x, y)

        adv_x = adv_x.cpu().numpy()
        return adv_x

    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs = self.classifier._model(x)[-1]
            y = self.classifier.reduce_labels(outputs)
        return y

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.classifier._model(im)[-1]

        g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
        grad_mask = torch.zeros_like(y)
        for counter in range(y.shape[-1]):
            zero_gradients(im)
            grad_mask[:, counter] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.data

        g2 = torch.transpose(g2, 0, 1).detach()
        y2 = y.detach()
        df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = torch.arange(imgs.shape[0])
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.classifier._model(im)[-1]
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = diffy.sum()

        zero_gradients(im)
        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg

    def attack_single_run(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
    

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()

        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                t = torch.randn(x1.shape).to(self.device)
                x1 = im2 + (torch.min(res2,
                                      self.eps * torch.ones(res2.shape)
                                      .to(self.device)
                                      ).reshape([-1, *[1]*self.ndims])
                            ) * t / ((t ** 2)
                                     .view(t.shape[0], -1)
                                     .sum(dim=-1)
                                     .sqrt()
                                     .view(t.shape[0], *[1]*self.ndims)) * .5

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.steps:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)

                    dist1 = df.abs() / (1e-12 + (dg ** 2)
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1).sqrt())
                 
                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1)
                         .sum(dim=-1))
                    w = dg2.reshape([bs, -1])

                    d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)

                    a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                            .view(-1, *[1]*self.ndims)
                    a0 = torch.max(a0, 1e-8 * torch.ones(
                        a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device)),
                                      self.alpha_max * torch.ones(a1.shape)
                                      .to(self.device))
                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)

                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                                .view(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        
        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def attack_single_run_targeted(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)


        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()

        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        output = self.classifier._model(x)[-2]
        if self.multi_targeted:
            la_target = output.sort(dim=-1)[1][:, -self.target_class]
        else:
            la_target = self.target_class

        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        la_target2 = la_target[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:   
                t = torch.randn(x1.shape).to(self.device)
                x1 = im2 + (torch.min(res2,
                                      self.eps * torch.ones(res2.shape)
                                      .to(self.device)
                                      ).reshape([-1, *[1]*self.ndims])
                                ) * t / ((t ** 2)
                                     .view(t.shape[0], -1)
                                     .sum(dim=-1)
                                     .sqrt()
                                     .view(t.shape[0], *[1]*self.ndims)) * .5

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.steps:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch_targeted(
                        x1, la2, la_target2)
                    
                    dist1 = df.abs() / (1e-12 + (dg ** 2)
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1).sqrt())
                   
                    ind = dist1.min(dim=1)[1]

                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1)
                         .sum(dim=-1))
                    w = dg2.reshape([bs, -1])

                    d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                
                    a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                            .view(-1, *[1]*self.ndims)
                    a0 = torch.max(a0, 1e-8 * torch.ones(
                        a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device)),
                                      self.alpha_max * torch.ones(a1.shape)
                                      .to(self.device))
                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                       
                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                                .view(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
   
        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        adv = x.clone()
        with torch.no_grad():

            outputs = self.classifier._model(x)[-1]
            acc = self.classifier.reduce_labels(outputs) == y

            def inner_perturb(targeted, acc):
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()  # nopep8

                        if targeted:
                            adv_curr = self.attack_single_run_targeted(
                                x_to_fool, y_to_fool, use_rand_start=(counter > 0))
                        else:
                            adv_curr = self.attack_single_run(
                                x_to_fool, y_to_fool, use_rand_start=(counter > 0))

                        outputs = self.classifier._model(adv_curr)[-1]
                        acc_curr = self.classifier.reduce_labels(outputs) == y_to_fool

                        res = ((x_to_fool - adv_curr)**2).view(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()  # nopep8
                        
                        acc_curr = torch.max(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

            inner_perturb(targeted=False, acc=acc)
        return adv


def projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1
    w.mul_(ind2.unsqueeze(1))
    c.mul_(ind2)

    r = torch.max(t / w, (t - 1) / w).clamp(min=-1e12, max=1e12)
    r.masked_fill_(w.abs() < 1e-8, 1e12)
    r[r == -1e12] *= -1
    rs, indr = torch.sort(r, dim=1)
    rs2 = F.pad(rs[:, 1:], (0, 1))
    rs.masked_fill_(rs == 1e12, 0)
    rs2.masked_fill_(rs2 == 1e12, 0)

    w3s = (w ** 2).gather(1, indr)
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r * w)
    d.mul_((w.abs() > 1e-8).float())
    s = torch.cat(
        (-w5 * rs[:, 0:1], torch.cumsum((-rs2 + rs) * ws, dim=1) - w5 * rs[:, 0:1]), 1)

    c4 = s[:, 0] + c < 0
    c3 = (d * w).sum(dim=1) + c > 0
    c2 = ~(c4 | c3)

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) + c_ > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb = lb.long()

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -alpha.unsqueeze(-1) * w[c4]

    if c2.any():
        alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
        alpha[ws[c2, lb] == 0] = 0
        c5 = (alpha.unsqueeze(-1) > r[c2]).float()
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).float()

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
