import copy
import numpy as np
from torch.autograd import Variable
import torch
import scipy.stats as st
from scipy import ndimage

import pdb


class TIDIM_Attack(object):
    def __init__(self, model, 
                       decay_factor=1, prob=0.5,
                       epsilon=0.3, steps=40, step_size=0.01, 
                       image_resize=330,
                       random_start=False):
        """
        Paper link: https://arxiv.org/pdf/1803.06978.pdf
        """

        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.rand = random_start
        self.model = copy.deepcopy(model)
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.decay_factor = decay_factor
        self.prob = prob
        self.image_resize = image_resize

        kernel = self.gkern(15, 3).astype(np.float32)
        self.stack_kernel = np.stack([kernel, kernel, kernel])

    def __call__(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat_np = X_nat.numpy()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.eval()
        if self.rand:
            X = X_nat_np + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat_np.shape).astype('float32')
        else:
            X = np.copy(X_nat_np)
        
        momentum = 0
        for _ in range(self.steps):
            X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
            y_var = y.cuda()

            rnd = np.random.rand()
            if rnd < self.prob:
                transformer = _tranform_resize_padding(X.shape[-2], X.shape[-1], self.image_resize, resize_back=True)
                X_trans_var = transformer(X_var)
            else:
                X_trans_var = X_var

            scores = self.model(X_trans_var)
            
            loss = self.loss_fn(scores, y_var)
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            grad = self.depthwise_conv2d(grad, self.stack_kernel)
            X_var.grad.zero_()

            velocity = grad / np.mean(np.absolute(grad))
            momentum = self.decay_factor * momentum + velocity

            X += self.step_size * np.sign(momentum)
            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range
        return torch.from_numpy(X)

    @staticmethod
    def gkern(kernlen, nsig):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    @staticmethod
    def depthwise_conv2d(in1, stack_kernel):
        ret = []
        for temp_in in in1:
            # numpy convolve operates differently to CNN conv, 
            # however they are the same when keernel is symetric.
            temp_out = ndimage.convolve(temp_in, stack_kernel, mode='constant')
            ret.append(temp_out)
        ret = np.array(ret)
        return ret


class _tranform_resize_padding(torch.nn.Module):
    def __init__(self, image_h, image_w, image_resize, resize_back=False):
        super(_tranform_resize_padding, self).__init__()
        self.shape = [image_h, image_w]
        self.image_resize = image_resize
        self.resize_back = resize_back

    def __call__(self, input_tensor):
        assert self.shape[0] < self.image_resize and self.shape[1] < self.image_resize
        rnd = np.random.randint(self.shape[1], self.image_resize)
        input_upsample = torch.nn.functional.interpolate(input_tensor, size=(rnd, rnd), mode='nearest')
        h_rem = self.image_resize - rnd
        w_rem = self.image_resize - rnd
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padder = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.0)
        input_padded = padder(input_upsample)
        if self.resize_back:
            input_padded_resize = torch.nn.functional.interpolate(input_padded, size=self.shape, mode='nearest')
            return input_padded_resize
        else:
            return input_padded
