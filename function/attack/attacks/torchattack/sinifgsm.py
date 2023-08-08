import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class SINIFGSM(EvasionAttack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        classifier 
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = SINIFGSM(classifier, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack.generate(x, y)

    """

    def __init__(self, classifier, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5):
   
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
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
        
        # if self.targeted:
        #     target_labels = self.get_target_label(x, y)

        momentum = torch.zeros_like(x).detach().to(self.device)

        adv_x = x.clone().detach()

        for _ in range(self.steps):
            adv_x.requires_grad = True
            nes_x = adv_x + self.decay*self.alpha*momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(x).detach().to(self.device)

            for i in torch.arange(self.m):
                nes_x = nes_x / torch.pow(2, i)
                outputs = self.classifier._model(nes_x)
                # Calculate loss
                # if self.targeted:
                #     cost = -loss(outputs, target_labels)
                # else:
                cost = self.classifier._loss(outputs[-1], y)
                adv_grad += torch.autograd.grad(cost, adv_x,
                                                retain_graph=True, create_graph=False)[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / \
                torch.mean(torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True)
            momentum = grad
            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x,
                                min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        adv_x = adv_x.cpu().numpy()
        return adv_x
