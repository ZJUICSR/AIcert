import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class MIFGSM(EvasionAttack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = MIFGSM(model, eps=8/255, steps=10, decay=1.0)
        >>> adv_images = attack.generate(x, y)

    """

    def __init__(self, classifier, eps=8/255, alpha=2/255, steps=10, decay=1.0):

        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
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
        #     target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(x).detach().to(self.device)

        adv_x = x.clone()

        for _ in range(self.steps):
            adv_x.requires_grad = True
           
            # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            outputs = self.classifier._model(adv_x)
            cost = self.classifier._loss(outputs[-1], y)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_x,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad),
                                     dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_x = adv_x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x - x,
                                min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        adv_x = adv_x.cpu().numpy()
        return adv_x
