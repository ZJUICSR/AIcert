import torch
import torch.nn as nn
from function.fairness.tabular.fairness_datasets import FairnessDataset
from function.fairness.tabular.models.models import Net
import numpy as np
from function.fairness.tabular.debias.inprocess.classifier import Classifier
from numpy.random import beta

class FMClassifier(Classifier):

    def __init__(self, input_shape, sensitive=None, device=None, method='mixup', mode='dp', lam=0.5):
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)
        self.method = method
        self.lam = lam
        self.mode = mode

        assert method in ['mixup', 'GapReg']
        assert mode in ['dp', 'eo']

    # def get_model(self, input_shape):
        # return Net(input_shape, 1, 50) # domain independent classifier requires an prediction for each domain

    def loss(self, x, y, z=None, w=None):
        if w is None:
            w = torch.ones(x.shape[0]).to(self.device)
        if self.mode == 'dp':
            return self.loss_dp(x, y, z, w)
        elif self.mode == 'eo':
            return self.loss_eo(x, y, z, w)

    def sample_batch(self, itr, batch_size, X, Y, Z, W):
        # dp requires number of z0 and z1 to be the same
        # eo requires number of z0y0 z0y1 z1y0 z1y1 to be the same
        batch_size = batch_size//2 if self.mode == 'dp' else batch_size//4
        idxs = torch.tensor([i for i in range(X.shape[0])])
        sel = []
        for i in range(2):
            if self.mode == 'dp':
                idx = idxs[Z==i]
                choice = torch.randint(0, idx.shape[0], (batch_size,))
                idx = idx[choice]
                sel.append(idx)
            else:
                for j in range(2):
                    idx = idxs[Z==i & Y==j]
                    choice = torch.randint(0, idx.shape[0], (batch_size,))
                    idx = idx[choice]
                    sel.append(idx)
                    
        sel = torch.cat(sel)
        x=X[sel].float().to(self.device)
        y=Y[sel].float().to(self.device)
        z=Z[sel].float().to(self.device)
        w=W[sel].float().to(self.device)
        return x, y, z, w
        
    
    def loss_dp(self, x, y, z, w):
        x_z_0 = x[z==0]
        x_z_1 = x[z==1]
        
        if self.method == 'mixup' and x_z_0.shape[0] == x_z_1.shape[0]: # skip when validating
            alpha = 1
            gamma = beta(alpha, alpha)
            x_z_mix = x_z_0 * gamma + x_z_1 * (1-gamma)
            x_z_mix.requires_grad_(True)
            y_hat = self.model(x_z_mix)

            # gradient regularization
            gradx = torch.autograd.grad(y_hat.sum(), x_z_mix, create_graph=True)[0]

            x_dist = x_z_1-x_z_0
            grad_inn = (gradx * x_dist).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = torch.abs(E_grad)
        elif self.method == 'GapReg':
            # Gap Regularization
            yh_z_0 = self.model(x_z_0)
            yh_z_1 = self.model(x_z_1)
            loss_reg = torch.abs(yh_z_0.mean()-yh_z_1.mean())
        else:
            loss_reg = 0
        
        
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)*w
        loss = loss + self.lam * loss_reg
        return loss, y_hat
    
    def loss_eo(self, x, y, z, w):
        x_z_0 = x[z==0]
        x_z_1 = x[z==1]
        
        # separate class
        n_z_0, n_z_1 = x_z_0.shape[0], x_z_1.shape[0]
        x_z_0_ = [x_z_0[:n_z_0//2], x_z_0[n_z_0//2:]]
        x_z_1_ = [x_z_1[:n_z_1//2], x_z_1[n_z_1//2:]]

        if self.method == 'mixup'and x_z_0.shape[0] == x_z_1.shape[0]:
            loss_reg = 0
            alpha = 1
            for i in range(2):
                gamma = beta(alpha, alpha)
                batch_x_0_i = x_z_0_[i]
                batch_x_1_i = x_z_1_[i]

                batch_x_mix = batch_x_0_i * gamma + batch_x_1_i * (1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)
                output = self.model(batch_x_mix)

                 # gradient regularization
                gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]
                batch_x_d = batch_x_1_i - batch_x_0_i
                grad_inn = (gradx * batch_x_d).sum(1)
                loss_reg += torch.abs(grad_inn.mean())

        elif self.method == "GapReg":
            loss_reg = 0
            for i in range(2):
                batch_x_0_i = x_z_0_[i]
                batch_x_1_i = x_z_1_[i]

                output_0 = self.model(batch_x_0_i)
                output_1 = self.model(batch_x_1_i)
                loss_reg += torch.abs(output_0.mean() - output_1.mean())
        else:
            # ERM
            loss_reg = 0

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)*w
        loss = loss + self.lam * loss_reg
        return loss, y_hat

    def predict(self, x, z):
        self.model.eval()
        y_hat = self.model(x)
        # y_hat = torch.unsqueeze(y_hat,dim=-1)
        return y_hat