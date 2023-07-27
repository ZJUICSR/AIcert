import torch
import torch.nn as nn
import numpy as np

class FFGSM():
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 10/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8/255, alpha=10/255):
        self.eps = eps
        self.alpha = alpha
        self.model = model
        self.device = next(model.parameters()).device

    def generate(self, x : np.ndarray, y : np.ndarray):
        r"""
        Overridden.
        """
        images = torch.from_numpy(x)
        labels = torch.from_numpy(y)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images + torch.randn_like(images).uniform_(-self.eps, self.eps)  # nopep8
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.model(adv_images)

        # Calculate loss
        # if self.targeted:
        #     cost = -loss(outputs, target_labels)
        # else:
        cost = loss(outputs, labels.long())

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
