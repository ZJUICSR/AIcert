import torch
import torch.nn as nn
from function.fairness.tabular.fairness_datasets import FairnessDataset
from function.fairness.tabular.models.models import Net
import math
import numpy as np
import copy
from . import Classifier

# domain independent training requires change of the model outpu
class FADClassifier(Classifier):
    def __init__(self, input_shape, sensitive=None, device=None):
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)
        self.criterion_bias = nn.CrossEntropyLoss(reduction='none')

    def get_model(self, input_shape):
        return Net(input_shape, 1, 50) # domain independent classifier requires an prediction for each domain

    def loss(self, x, y, z, w=None):
        if w is None:
            w = torch.ones(x.shape[0]).to(self.device)
        y_hat, z_hat = self.model(x)
        y_hat, y = torch.squeeze(y_hat, dim=1), torch.squeeze(y, dim=1)
        loss = self.criterion(y_hat, y) + \
            self.criterion_bias(z_hat, z.long())
        return loss*w, y_hat

    def predict(self, x, z):
        self.model.eval()
        y_hat, z_hat = self.model(x)
        # y_hat = torch.unsqueeze(y_hat,dim=-1)
        return y_hat
