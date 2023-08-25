import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class EOTPGD(EvasionAttack):
    r"""
    Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
    [https://arxiv.org/abs/1907.00895]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        eot_iter (int) : number of models to estimate the mean gradient. (Default: 2)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = EOTPGD(classifier, eps=8/255, alpha=2/255, steps=10, eot_iter=2)
        >>> adv_images = attack.generate(images, labels)

    """

    def __init__(self, classifier, eps=8/255, alpha=2/255, steps=10,
                 eot_iter=2, random_start=True):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.eot_iter = eot_iter
        self.random_start = random_start
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
        adv_x = x.clone()

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        if self.random_start:
            # Starting at a uniformly random point
            adv_x = adv_x + torch.empty_like(adv_x).uniform_(-self.eps, self.eps)  # nopep8
            adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_x)

            adv_x.requires_grad = True

            for j in range(self.eot_iter):
                outputs = self.classifier._model(adv_x)

                # Calculate loss
                # if self.targeted:
                #     cost = -loss(outputs, target_labels)
                # else:
                cost = self.classifier._loss(outputs[-1], y)

                # Update adversarial x
                grad += torch.autograd.grad(cost, adv_x,
                                            retain_graph=False,
                                            create_graph=False)[0]

            # (grad/self.eot_iter).sign() == grad.sign()
            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x,
                                min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        adv_x = adv_x.cpu().numpy()

        return adv_x
