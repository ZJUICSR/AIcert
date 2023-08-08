import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class TPGD(EvasionAttack):
    r"""
    PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]

    Distance Measure : Linf

    Arguments:
        classifier 
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = TPGD(classifier, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack.generate(x)

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

        images = torch.from_numpy(x).to(self.device)
        labels = torch.from_numpy(y)

        logit_ori = self.classifier._model(images)[-1].detach()

        adv_images = images + 0.001*torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction='sum')

        for _ in range(self.steps):
            adv_images.requires_grad = True
            logit_adv = self.classifier._model(adv_images)[-1]

            # Calculate loss
            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        adv_images = adv_images.cpu().numpy()
        return adv_images
