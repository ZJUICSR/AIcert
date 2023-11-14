import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class RFGSM(EvasionAttack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    Distance Measure : Linf

    Arguments:
        classifier 
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.RFGSM(classifier, eps=8/255, alpha=2/255, steps=10)
        >>> adv_x = attack(x, y)
    """

    def __init__(self, classifier, eps=8/255, alpha=2/255, steps=10):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
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

        adv_x = x + self.alpha*torch.randn_like(x).sign()
        adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_x.requires_grad = True
            outputs = self.classifier._model(adv_x)

            # Calculate loss
            cost = self.classifier._loss(outputs[-1], y)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]
            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x,
                                min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        adv_x = adv_x.cpu().numpy()
        return adv_x
