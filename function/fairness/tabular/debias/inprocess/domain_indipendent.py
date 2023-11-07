import torch
import torch.nn as nn
from fairness_datasets import FairnessDataset
from models.models import Net
import math
import numpy as np
import copy
from . import Classifier

# domain independent training requires change of the model output 
class DomainIndependentClassifier(Classifier):
    def __init__(self, input_shape, sensitive=None, device=None):
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)

    def loss(self, x, y, z, w=None):
        if w is None:
            w = torch.ones(x.shape[0]).to(self.device)
        predict_prob = self.model(x)
        yh = z * predict_prob[:, 0] + (1-z) * predict_prob[:, 1]
        yh = torch.unsqueeze(yh,dim=-1)
        loss = self.criterion(yh, y)*w
        return loss, yh

    def get_model(self, input_shape):
        return Net(input_shape, output_shape=2, grl_lambda=0) # domain independent classifier requires an prediction for each domain

    def predict(self, x, z):
        self.model.eval()
        predict_prob = self.model(x)
        yh = z * predict_prob[:, 0] + (1-z) * predict_prob[:, 1]
        yh = torch.unsqueeze(yh,dim=-1)
        return yh

