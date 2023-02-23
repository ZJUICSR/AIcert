import numpy as np

import torch

from adv.base import Base
from .deepfool import DeepFool


class SparseFool(Base):
    r"""
    Attack in the paper 'SparseFool: a few pixels make a big difference'
    [https://arxiv.org/abs/1811.02248]
    
    Modified from "https://github.com/LTS4/SparseFool/"
    
    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFAULT: 20)
        lam (float): parameter for scaling DeepFool noise. (DEFAULT: 3)
        overshoot (float): parameter for enhancing the noise. (DEFALUT: 0.02)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, steps=20, lam=3, overshoot=0.02):
        super(SparseFool, self).__init__("SparseFool", model)
        self.steps = steps
        self.lam = lam
        self.overshoot = overshoot
        self.deepfool = DeepFool(model)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        curr_steps = 0
        
        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)
        
        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                image = images[idx:idx+1]
                label = labels[idx:idx+1]
                adv_image = adv_images[idx]
                
                fs = self.model(adv_image)[0]
                _, pre = torch.max(fs, dim=0)
                if pre != label:
                    correct[idx] = False
                    continue
                
                adv_image, target_label = self.deepfool(adv_image, label,
                                                        return_target_labels=True)
                adv_image = image + self.lam*(adv_image - image)

                adv_image.requires_grad = True
                fs = self.model(adv_image)[0]
                _, pre = torch.max(fs, dim=0)

                if pre == label: pre = target_label
                
                cost = fs[pre] - fs[label]
                grad = torch.autograd.grad(cost, adv_image,
                                           retain_graph=False, create_graph=False)[0]
                grad = grad / grad.norm()
                
                adv_image = self._linear_solver(image, grad, adv_image)
                adv_image = image + (1+self.overshoot)*(adv_image - image)
                adv_images[idx] = torch.clamp(adv_image, min=self.bounds[0], max=self.bounds[1]).detach()

            curr_steps += 1
           
        adv_images = torch.cat(adv_images).detach()
               
        return adv_images

    def _linear_solver(self, x_0, coord_vec, boundary_point):
        input_shape = x_0.size()

        plane_normal = coord_vec.clone().detach().view(-1)
        plane_point = plane_normal.clone().detach().view(-1)

        x_i = x_0.clone().detach()

        f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
        sign_true = f_k.sign().item()

        beta = 0.001 * sign_true
        current_sign = sign_true

        while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:

            f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

            pert = f_k.abs() / coord_vec.abs().max()

            mask = torch.zeros_like(coord_vec)
            mask[np.unravel_index(torch.argmax(coord_vec.abs()).cpu(), input_shape)] = 1.

            r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

            x_i = x_i + r_i
            x_i = torch.clamp(x_i, min=0, max=1)

            f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
            current_sign = f_k.sign().item()

            coord_vec[r_i != 0] = 0

        return x_i
