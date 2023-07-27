import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack

class DIFGSM(EvasionAttack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        classifier : PyTorchClassifier
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 10)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = DIFGSM(classifer, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_x= attack.generate(x, y)

    """

    def __init__(self, classifier, eps=8/255, alpha=2/255, steps=10, decay=0.0,
                 resize_rate=0.9, diversity_prob=0.5, random_start=False):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.classifier = classifier
        self.device = classifier.device


    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x


    def generate(self, x : np.ndarray, y : Optional[np.ndarray] = None):
        r"""
        Overridden.
        """
        if y is None:
            y = self.classifier.predict(x)
        
        if len(y.shape) == 2:
            y = self.classifier.reduce_labels(y)

        y = torch.from_numpy(y).to(self.device)
        x = torch.from_numpy(x).to(self.device)

        momentum = torch.zeros_like(x)

        adv_x = x.clone()

         # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        if self.random_start:
            # Starting at a uniformly random point
            adv_x = adv_x + torch.empty_like(adv_x).uniform_(-self.eps, self.eps)
            adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for _ in range(self.steps):
            
            adv_x.requires_grad = True

            # Calculate loss
            outputs = self.classifier._model(self.input_diversity(adv_x))
            cost = self.classifier._loss(outputs[-1], y)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]
           
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()
            
        adv_x = adv_x.cpu().numpy()

        return adv_x
