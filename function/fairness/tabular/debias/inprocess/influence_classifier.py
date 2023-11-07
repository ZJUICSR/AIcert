import torch
import torch.nn as nn
from function.fairness.tabular.fairness_datasets import FairnessDataset
from function.fairness.tabular.models.models import Net, Net2
import math
import numpy as np
import copy
from function.fairness.tabular.debias.inprocess.classifier import Classifier
from tqdm import tqdm
from typing import Sequence, Tuple
from numpy import linalg as la
from scipy.linalg import cho_solve, cho_factor

def check_positive_definite(M: np.ndarray) -> bool:
    """ Returns true when input is positive-definite, via Cholesky """

    try:
        _ = la.cholesky(M)
        return True
    except la.LinAlgError:
        return False

def nearest_pd(M: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (M + M.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if check_positive_definite(A3):
        return A3

    spacing = np.spacing(la.norm(M))

    """
    The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    `spacing` will, for Gaussian random matrixes of small dimension, be on
    othe order of 1e-16. In practice, both ways converge, as the unit test
    below suggests.
    """

    I = np.eye(M.shape[0])
    k = 1
    while not check_positive_definite(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3



class IFClassifier(Classifier):
    
    @staticmethod
    def set_sample_weight(n: int, sample_weight: np.ndarray or Sequence[float] = None) -> np.ndarray:
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            if isinstance(sample_weight, np.ndarray):
                assert sample_weight.shape[0] == n
            elif isinstance(sample_weight, (list, tuple)):
                assert len(sample_weight) == n
                sample_weight = np.array(sample_weight)
            else:
                raise TypeError

            assert min(sample_weight) >= 0.
            assert max(sample_weight) <= 2.

        return sample_weight
    
    @staticmethod
    def check_pos_def(M: np.ndarray) -> bool:
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def
    
    @staticmethod
    def get_inv_hvp(hessian: np.ndarray, vectors: np.ndarray, cho: bool = True) -> np.ndarray:
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        else:
            hess_inv = np.linalg.inv(hessian)
            return hess_inv.dot(vectors.T)
    
    @staticmethod
    def log_loss(y, y_hat, w=None, eps=1e-16):
        return - y * torch.log(y_hat + eps) - (1. - y) * torch.log(1. - y_hat + eps)
    
    def __init__(self, input_shape, sensitive=None, device=None):
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)
        
        self.criterion = self.log_loss
        # self.criterion = nn.MSELoss(reduction='none')
        
    def get_model(self, input_shape):
        return Net2(input_shape, 1, 0)
    

    
    def loss(self, x, y, z=None, w=None):
        eps = 1e-16
        if w is None:
            w = torch.ones(x.shape[0]).to(self.device)
        y_hat = self.predict(x, z)
        log_loss = self.criterion(y, y_hat)
        try:
            log_loss = torch.matmul(w, log_loss)
        except:
            log_loss = w.mul(log_loss)
        # if l2_reg:
        #     weight = self.model.weight
        #     log_loss += torch.linalg.norm(weight, ord=2).div(2.).mul(self.l2_reg)

        return log_loss, y_hat
    
    def loss_np(self, x, y, z=None, w=None):
        x, y = map(torch.from_numpy, (x, y))
        x, y = x.float().to(self.device), y.float().to(self.device)
        z = None if z is None else torch.from_numpy(z).float().to(self.device)
        w = None if w is None else torch.from_numpy(w).to(self.device)
        loss, y_hat = self.loss(x, y, z, w)
        # if l2_reg:
        #     weight = self.model.weight.flatten()
        #     weight = weight.detach().cpu().numpy()
        #     log_loss += self.l2_reg * np.linalg.norm(weight, ord=2) / 2.

        return loss.detach().cpu().numpy()

    def grad(self, x, y, w=None, l2_reg=False):

        N = x.shape[0]
        w = self.set_sample_weight(N, w)

        x = torch.from_numpy(x).to(torch.float).to(self.device)
        y = torch.from_numpy(y).to(torch.float).to(self.device)
        w = torch.from_numpy(w).float().to(self.device)

        weighted_indiv_grad = np.zeros((N, self.model.num_parameters))
        pred = self.model(x)
        # loss, _ = self.loss(x, y, w=w)
        for i, (single_pred, single_y, single_weight) in enumerate(
                tqdm(zip(pred, y, w), total=N, desc="Computing individual first-order gradient")):
            indiv_loss = self.criterion(single_y, single_pred)*single_weight
            indiv_grad = torch.autograd.grad(indiv_loss, self.model.parameters(), retain_graph=True)
            indiv_grad = torch.cat([x.flatten() for x in indiv_grad], dim=0).detach().cpu().numpy()
            weighted_indiv_grad[i] = indiv_grad

        total_grad = np.sum(weighted_indiv_grad, axis=0)
        return total_grad, weighted_indiv_grad
    
    def grad_pred(self, x, sample_weight=None):
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """
        N = x.shape[0]
        sample_weight = self.set_sample_weight(N, sample_weight)

        x = torch.from_numpy(x).to(torch.float).to(self.device)
        sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        weighted_indiv_grad = np.zeros((N, self.model.num_parameters))
        pred = self.model(x)
        for i, (single_pred,  single_weight) in enumerate(
                tqdm(zip(pred, sample_weight), total=N, desc="Computing individual first-order gradient")):
            # indiv_loss = self.log_loss_tensor(single_y, single_pred, single_weight) # weighted loss for an individual sample
            indiv_grad = torch.autograd.grad(single_pred*single_weight, self.model.parameters(), retain_graph=True) # ! compute gradient of y_hat
            indiv_grad = torch.cat([x.flatten() for x in indiv_grad], dim=0).detach().cpu().numpy()
            weighted_indiv_grad[i] = indiv_grad

        total_grad = np.sum(weighted_indiv_grad, axis=0) # (num_sample, num_parameter) -> (num_parameter,)
        reg_grad = self.l2_reg * self.model.weight
        reg_grad = reg_grad.detach().cpu().numpy() # gradients of training individual loss w.r.t. parameters


        return total_grad, weighted_indiv_grad
    
    def hess(self, x, y, w=None, check_pos_def=True) -> np.ndarray:
        """
        Compute hessian matrix for the whole training set
        """

        w = self.set_sample_weight(x.shape[0], w)

        x = torch.from_numpy(x).to(torch.float).to(self.device)
        y = torch.from_numpy(y).to(torch.float).to(self.device)
        w = torch.from_numpy(w).float().to(self.device)

        pred = self.model(x)
        loss = self.criterion(y, pred)*w
        hess_wo_reg = torch.zeros((self.model.num_parameters, self.model.num_parameters)).to(self.device)

        grad = torch.autograd.grad(outputs=loss.mean(), inputs=self.model.parameters(), create_graph=True)
        grad = torch.cat([x.flatten() for x in grad], dim=0)

        for i, g in enumerate(tqdm(grad, total=self.model.num_parameters, desc="Computing second-order gradient")):
            second_order_grad = torch.autograd.grad(outputs=g, inputs=self.model.parameters(), retain_graph=True)
            second_order_grad = torch.cat([x.flatten() for x in second_order_grad], dim=0)
            hess_wo_reg[i, :] = second_order_grad

        hess_wo_reg = hess_wo_reg.detach().cpu().numpy()
        # reg_hess = self.l2_reg * np.eye(self.model.num_parameters)
        reg_hess = 2.26220 * np.eye(self.model.num_parameters)

        total_hess_w_reg = hess_wo_reg + reg_hess
        total_hess_w_reg = (total_hess_w_reg + total_hess_w_reg.T) / 2.

        if check_pos_def:
            pd_flag = self.check_pos_def(total_hess_w_reg)
            if not pd_flag:
                print("Converting hessian to nearest positive-definite matrix")
                total_hess_w_reg = nearest_pd(total_hess_w_reg)
                self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg

    def grad_pred(self, x, sample_weight=None):
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)
        emb = self.emb(x)
        indiv_grad = emb * (pred * (1 - pred)).reshape(-1, 1)
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        total_grad = np.sum(weighted_indiv_grad, axis=0)

    def predict(self, x, z):
        self.model.eval()
        yh = self.model(x)
        # yh = torch.unsqueeze(yh,dim=-1)
        return yh

    def predicted_dataset(self, dataset):
        available = list(dataset.privileged.keys())
        if self.sensitive is None:
            self.sensitive = available[0]
        if self.sensitive not in available:
            raise ValueError(f"invalid sensitive value: \'{self.sensitive}\' is not in the dataset.")
        idx = available.index(self.sensitive)
        x = torch.from_numpy(np.array(dataset.X)).float().to(device=self.device)
        z = torch.from_numpy(np.array(dataset.Z)[:, idx]).float().to(device=self.device)
        yh = self.predict(x, z).detach().cpu().numpy()
        predicted_dataset = copy.deepcopy(dataset)
        predicted_dataset.Y = yh
        return predicted_dataset