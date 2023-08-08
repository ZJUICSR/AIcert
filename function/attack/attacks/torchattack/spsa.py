import torch
from torch.nn.modules.loss import _Loss
import numpy as np
from typing import Optional
from function.attack.attacks.attack import EvasionAttack


class MarginalLoss(_Loss):
    def forward(self, logits, targets):
        assert logits.shape[-1] >= 2
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets.long()]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets,
            top_logits[..., 1],
            top_logits[..., 0],
        )

        loss = max_nontarget_logits - target_logits
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError("unknown reduction: '%s'" % (self.recution,))

        return loss


class SPSA(EvasionAttack):
    r"""
    SPSA in the paper 'Adversarial Risk and the Dangers of Evaluating Against Weak Attacks'
    [https://arxiv.org/abs/1802.05666]

    Distance Measure : Linf

    Arguments:
        classifier 
        eps (float): maximum perturbation. (Default: 8/255)
        delta (float): scaling parameter of SPSA.
        lr (float): the learning rate of the `Adam` optimizer.
        nb_iter (int): number of iterations of the attack.
        nb_sample (int): number of samples for SPSA gradient approximation.
        max_batch_size (int): maximum batch size to be evaluated at once.

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = SPSA(classifier, eps=0.3)
        >>> adv_images = attack.generate(x, y)

    """

    def __init__(self, classifier, eps=0.3, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64):
  
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size
        self.loss_fn = MarginalLoss(reduction="none")
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
        labels = torch.from_numpy(y).to(self.device)

        adv_images = self.spsa_perturb(images, labels)

        adv_images = adv_images.cpu().numpy()
        return adv_images

    def loss(self, *args):
        return -self.loss_fn(*args)

    def linf_clamp_(self, dx, x, eps):
        """Clamps perturbation `dx` to fit L_inf norm and image bounds.

        Limit the L_inf norm of `dx` to be <= `eps`, and the bounds of `x + dx`
        to be in `[clip_min, clip_max]`.

        Return: the clamped perturbation `dx`.
        """

        # dx_clamped = self.batch_clamp(eps, dx)
        dx_clamped = torch.clamp(dx, min=-eps, max=eps)
        # x_adv = self.clamp(x + dx_clamped, clip_min, clip_max)
        x_adv = torch.clamp(x + dx_clamped, min=0, max=1)
        # `dx` is changed *inplace* so the optimizer will keep
        # tracking it. the simplest mechanism for inplace was
        # adding the difference between the new value `x_adv - x`
        # and the old value `dx`.
        dx += x_adv - x - dx
        return dx

    def _get_batch_sizes(self, n, max_batch_size):
        batches = [max_batch_size for _ in range(n // max_batch_size)]
        if n % max_batch_size > 0:
            batches.append(n % max_batch_size)
        return batches

    @torch.no_grad()
    def spsa_grad(self, images, labels, delta, nb_sample, max_batch_size):
        """Uses SPSA method to apprixmate gradient w.r.t `x`.

        Use the SPSA method to approximate the gradient of `loss(predict(x), y)`
        with respect to `x`, based on the nonce `v`.

        Return the approximated gradient of `loss_fn(predict(x), y)` with respect to `x`.
        """

        grad = torch.zeros_like(images)
        images = torch.unsqueeze(images, 0)
        labels = torch.unsqueeze(labels, 0)

        def f(xvar, yvar):
            return self.loss(self.classifier._model(xvar)[-1], yvar)

        images = images.expand(max_batch_size, *images.shape[1:]).contiguous()
        labels = labels.expand(max_batch_size, *labels.shape[1:]).contiguous()

        v = torch.empty_like(images[:, :1, ...])
        for batch_size in self._get_batch_sizes(nb_sample, max_batch_size):
            x_ = images[:batch_size]
            y_ = labels[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *images.shape[2:])
            y_ = y_.view(-1, *labels.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = f(x_+delta*v_, y_) - f(x_-delta*v_, y_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2.*delta*v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_

        grad /= nb_sample
        return grad

    def spsa_perturb(self, x, y):
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(
                x + dx, y, self.delta, self.nb_sample, self.max_batch_size)
            optimizer.step()
            dx = self.linf_clamp_(dx, x, self.eps)

        x_adv = x + dx
        return x_adv
