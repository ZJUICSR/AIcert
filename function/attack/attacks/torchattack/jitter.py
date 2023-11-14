import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class Jitter(EvasionAttack):
    r"""
    Jitter in the paper 'Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks'
    [https://arxiv.org/abs/2105.10304]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = Jitter(model, eps=8/255, alpha=2/255, steps=10,
                 scale=10, std=0.1, random_start=True)
        >>> adv_images = attack.generate(x, y)

    """

    def __init__(self, classifier, eps=8/255, alpha=2/255, steps=10,
                 scale=10, std=0.1, random_start=True):
  
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.scale = scale
        self.std = std
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

        y = torch.from_numpy(y).to(self.device)

        x = torch.from_numpy(x).to(self.device)

        loss = nn.MSELoss(reduction='none')

        adv_x = x.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_x = adv_x + \
                torch.empty_like(adv_x).uniform_(-self.eps, self.eps)
            adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_x.requires_grad = True

            logits = self.classifier._model(adv_x)[-1]
            
            _, pre = torch.max(logits, dim=1)
            # outputs = self.classifier._model(adv_x)[-1]
            # pre = self.classifier.reduce_labels(outputs)
            wrong = (pre != y)

            norm_z = torch.norm(logits, p=float('inf'), dim=1, keepdim=True)
            hat_z = nn.Softmax(dim=1)(self.scale*logits/norm_z)

            if self.std != 0:
                hat_z = hat_z + self.std*torch.randn_like(hat_z)

            # Calculate loss
            # if self.targeted:
            #     target_Y = F.one_hot(
            #         target_labels, num_classes=logits.shape[-1]).float()
            #     cost = -loss(hat_z, target_Y).mean(dim=1)
            # else:
            Y = F.one_hot(y, num_classes=logits.shape[-1]).float()
            cost = loss(hat_z, Y).mean(dim=1)

            norm_r = torch.norm((adv_x - x), p=float('inf'), dim=[1, 2, 3])  # nopep8
            nonzero_r = (norm_r != 0)
            cost[wrong*nonzero_r] /= norm_r[wrong*nonzero_r]

            cost = cost.mean()

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x-x, min=-self.eps, max=self.eps)  # nopep8
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        adv_x = adv_x.cpu().numpy()
        return adv_x
