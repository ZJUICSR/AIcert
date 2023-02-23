import numpy as np

import torch
import torch.nn.functional as F

from argp.third_party.attacks.adv.base import Base
from ._differential_evolution import differential_evolution


class OnePixel(Base):
    r"""
    Attack in the paper 'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]
    
    Modified from "https://github.com/DebangLi/one-pixel-attack-pytorch/" and 
    "https://github.com/sarathknv/adversarial-examples-pytorch/blob/master/one_pixel_attack/"
    
    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        pixels (int): number of pixels to change (DEFAULT: 1)
        steps (int): number of steps. (DEFAULT: 75)
        popsize (int): population size, i.e. the number of candidate agents or "parents" in differential evolution (DEFAULT: 400)
        inf_batch (int): maximum batch size during inference (DEFAULT: 128)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.OnePixel(model, pixels=1, steps=75, popsize=400, inf_batch=128)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, pixels=1, steps=75, popsize=400, inf_batch=128):
        super(OnePixel, self).__init__("OnePixel", model)
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        batch_size, channel, height, width = images.shape
        
        bounds = [(0, height), (0, width)]+[(0, 1)]*channel
        bounds = bounds*self.pixels
        
        popmul = max(1, int(self.popsize/len(bounds)))
                     
        adv_images = []
        for idx in range(batch_size):
            image, label = images[idx:idx+1], labels[idx:idx+1]
            delta = differential_evolution(func=lambda delta: self._loss(image, label, delta),
                                           bounds=bounds,
                                           callback=lambda delta, convergence:\
                                                     self._attack_success(image, label, delta),
                                           maxiter=self.steps, popsize=popmul,
                                           init='random',
                                           recombination=1, atol=-1, 
                                           polish=False).x
            delta = np.split(delta, len(delta)/len(bounds))
            adv_image = self._perturb(image, delta)
            adv_images.append(adv_image)
        
        adv_images = torch.cat(adv_images)
        return adv_images
    
    def _loss(self, image, label, delta):
        adv_images = self._perturb(image, delta)  # Mutiple delta
        prob = self._get_prob(adv_images)[:, label]
        if (self._targeted == 1):
            return 1-prob  # If targeted, increase prob
        else:
            return prob  # If non-targeted, decrease prob
    
    def _attack_success(self, image, label, delta):
        adv_image = self._perturb(image, delta) # Single delta
        prob = self._get_prob(adv_image)
        pre = np.argmax(prob)
        if (self._targeted == 1) and (pre == label):
            return True
        elif (self._targeted == -1) and (pre != label):
            return True
        return False
    
    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.model(batch)
                outs.append(out)
        outs = torch.cat(outs)
        prob = F.softmax(outs, dim=1)
        return prob.detach().cpu().numpy()
    
    def _perturb(self, image, delta):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image]*num_delta, dim=0)
        for idx in range(num_delta):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                pos_x, pos_y = pixel[:2]
                channel_v = pixel[2:]
                for channel, v in enumerate(channel_v):
                    adv_images[idx, channel, int(pos_x), int(pos_y)] = v
        return adv_images
