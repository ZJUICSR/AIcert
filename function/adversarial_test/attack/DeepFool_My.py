# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torchattacks.attack import Attack
from tqdm import tqdm

class DeepFoolMy(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, steps=50, overshoot=0.02):
        super().__init__("DeepFoolMy", model)
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]
        # mask = self.mask.repeat(batch_size, 1, 1, 1)
        # eps = self.eps.repeat(batch_size, 1, 1, 1)
        # alpha = self.alpha.repeat(batch_size, 1, 1, 1)

        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        target_labels = labels.clone().detach().to(self.device)
        # curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        # while (True in correct) and (curr_steps < self.steps):
        for step in tqdm(range(self.steps), ncols=100, desc='Steps', leave=False):
            if True not in correct:
                break
            for idx in tqdm(range(batch_size), ncols=100, desc=f'step process {step + 1}', leave=False):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                # if self.mask is not None:
                #     adv_image = torch.where(mask[idx] > 0, adv_image, images[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            # curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        fs, _ = torch.sort(fs, descending=True)
        fs = fs[0: 10]
        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[0]
        w_0 = ws[0]

        wrong_classes = [i for i in range(len(fs)) if i != 0]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label else hat_L+1

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        # grad_num = 10
        # _, index_list = torch.sort(y, descending=True)
        # print(f'label={label}')
        # print(f'label={type(label)}')
        # print(f'index_list={index_list}')
        # print(f'index_list={type(index_list)}')
        # index_list = set(index_list[: grad_num]).add(label)
        # print(f'index={index_list}')
        for idx, y_element in enumerate(y):
            # if idx not in index_list:
            #     continue
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        # print(f'stack.shape={torch.stack(x_grads).shape}')
        # x_grads *= 100
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)


if __name__ == '__main__':
    pass

