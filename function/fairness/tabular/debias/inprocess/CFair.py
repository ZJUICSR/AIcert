import torch
import torch.nn as nn
from function.fairness.tabular.fairness_datasets import FairnessDataset
from function.fairness.tabular.models.models import CFairNet, CONFIGS
import math
import numpy as np
import copy
from . import Classifier
import torch.nn.functional as F
from function.fairness.tabular.metrics.dataset_metric import DatasetMetrics

# domain independent training requires change of the model outpu
class CFairClassifier(Classifier):
    def __init__(self, input_shape, mu=10, mode='cfair', sensitive=None, device=None):
        self.num_classes = 2
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)
        self.mu = mu
        self.mode = mode
        self.criterion_bias = F.nll_loss
        self.criterion = F.nll_loss
        

    def get_model(self, input_shape):
        c = copy.deepcopy(CONFIGS)
        c['num_classes'] = self.num_classes
        return CFairNet(input_shape, 2, 1, c) 
    
    def train(self, dataset: FairnessDataset, epochs=200, batch_size=None, thresh=0.5):
        # For reweighing purpose.
        idx = self.check_sens(dataset)
        dm = DatasetMetrics(dataset)
        train_y_1 = list(dm.favorable_rate(None).values())[idx]
        train_base_0 = list(dm.favorable_rate(False).values())[idx]
        train_base_1 = list(dm.favorable_rate(True).values())[idx]
        if self.mode == "cfair":
            self.reweight_target_tensor = torch.tensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1]).float().to(self.device)
        elif self.mode == "cfair-eo":
            self.reweight_target_tensor = torch.tensor([1.0, 1.0]).float().to(self.device)
        reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).float().to(self.device)
        reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).float().to(self.device)
        self.reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]
        # self.reweight_target_tensor = reweight_target_tensor
        return super().train(dataset, epochs, batch_size, thresh)

    def loss(self, x, y, z, w):
        y = torch.squeeze(y,dim=1)
        y_hat, z_hat = self.model(x, y)
        loss = self.criterion(y_hat, y.long(), weight=self.reweight_target_tensor, reduce=False) # + self.criterion_bias(z_hat, torch.unsqueeze(z, dim=-1))
        if w is not None:
            loss = loss * w
        loss = loss.mean()
        adv_loss = torch.mean(torch.stack([F.nll_loss(z_hat[j], z[y == j].long(), weight=self.reweight_attr_tensors[j]) for j in range(self.num_classes)]))
        loss += self.mu * adv_loss
        return loss, y_hat

    def predict(self, x, z):
        self.model.eval()
        y_hat = self.model.inference(x)
        # y_hat = torch.unsqueeze(y_hat,dim=-1)
        return y_hat
