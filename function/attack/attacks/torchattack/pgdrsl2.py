import torch
import torch.nn.functional as F
import numpy as np 
import copy
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class Noise():
    def __init__(self, noise_type, noise_sd):
        self.noise_type = noise_type
        self.noise_sd = noise_sd

    def __call__(self, img):
        if self.noise_type == "guassian":
            noise = torch.randn_like(img.float())*self.noise_sd
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(img.float()) - 0.5)*2*self.noise_sd
        return noise


class PGDRSL2(EvasionAttack):
    r"""
    PGD for randmized smoothing in the paper 'Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers'
    [https://arxiv.org/abs/1906.04584]
    Modification of the code from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/attacks.py

    Distance Measure : L2

    Arguments:
        classifier 
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        noise_type (str): guassian or uniform. (Default: guassian)
        noise_sd (float): standard deviation for normal distributio, or range for . (Default: 0.5)
        noise_batch_size (int): guassian or uniform. (Default: 5)
        batch_max (int): split data into small chunk if the total number of augmented data points, len(inputs)*noise_batch_size, are larger than batch_max, in case GPU memory is insufficient. (Default: 2048)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDRSL2(classifier, eps=1.0, alpha=0.2, steps=10, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048)
        >>> adv_x = attack(x, y)

    """

    def __init__(self, classifier, eps=1.0, alpha=0.2, steps=10, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048, eps_for_division=1e-10):

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.noise_func = Noise(noise_type, noise_sd)
        self.noise_batch_size = noise_batch_size
        self.eps_for_division = eps_for_division
        self.batch_max = batch_max
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

        if x.shape[0]*self.noise_batch_size > self.batch_max:
            x_list = []
            split_num = int(self.batch_max/self.noise_batch_size)
            x_split = torch.split(x, split_size_or_sections=split_num)
            y_split = torch.split(x, split_size_or_sections=split_num)
            for x_sub, y_sub in zip(x_split, y_split):
                x_adv = self._generate(x_sub, y_sub)
                x_list.append(x_adv)
            return torch.vstack(x_list)
        else:
            return self._generate(x, y)

    def _generate(self, x, y):
        r"""
        Overridden.
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        # expend the inputs over noise_batch_size
        shape = torch.Size([x.shape[0], self.noise_batch_size]) + x.shape[1:]  # nopep8
        inputs_exp = x.unsqueeze(1).expand(shape)
        inputs_exp = inputs_exp.reshape(torch.Size([-1]) + inputs_exp.shape[2:])  # nopep8

        data_batch_size = y.shape[0]
        delta = torch.zeros(
            (len(y), *inputs_exp.shape[1:]), requires_grad=True, device=self.device)
        delta_last = torch.zeros(
            (len(y), *inputs_exp.shape[1:]), requires_grad=False, device=self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(x, y)

        for _ in range(self.steps):
            delta.requires_grad = True
            # img_adv is the perturbed data for randmized smoothing
            # delta.repeat(1,self.noise_batch_size,1,1)
            img_adv = inputs_exp + delta.unsqueeze(1).repeat((1, self.noise_batch_size, 1, 1, 1)).view_as(inputs_exp)  # nopep8
            img_adv = torch.clamp(img_adv, min=0, max=1)

            noise_added = self.noise_func(img_adv.view(len(img_adv), -1))
            noise_added = noise_added.view(img_adv.shape)

            adv_noise = img_adv + noise_added
            logits = self.classifier._model(adv_noise)[-1]

            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = softmax.reshape(
                -1, self.noise_batch_size, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsoftmax, y) 

            grad = torch.autograd.grad(
                ce_loss, delta, retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(
                grad.view(data_batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(data_batch_size, 1, 1, 1)

            delta = delta_last + self.alpha*grad
            delta_norms = torch.norm(delta.view(
                data_batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
            delta_last.data = copy.deepcopy(delta.data)

        adv_x = torch.clamp(x + delta, min=0, max=1).detach()
        
        adv_x = adv_x.cpu().numpy()
        return adv_x
