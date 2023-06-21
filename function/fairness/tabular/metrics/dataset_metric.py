import sys
import os
sys.path.append(os.path.join(os.getcwd(),"function/fairness"))

from fairness_datasets import FairnessDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
from metrics.metric_utils import dic_operation
from metrics.metric_utils import *


# select out priviledged and unpriiviledged samples from dataset
class DatasetMetrics(): # binary label and binary group only

    # sensitive attributes and label are given by dataset
    def __init__(self, dataset: FairnessDataset, sensitive=[]):
        self.dataset = dataset
        self.priviledged = dataset.privileged
        self.favorable = dataset.favorable

        # extract vector of sensitive attribute and label
        self.z, self.y = np.array(dataset.Z), np.array(dataset.Y)
        self.w = dataset.weights

        # mask for privileged an unprivileged
        self.pri_mask = (self.z == 1)
        self.unp_mask = (self.z != 1)

        # statistics of groups
        self.total_num = np.sum(self.w)
        self.pri_num = np.sum(self.pri_mask * self.w, axis=0)
        self.unp_num = np.sum(self.unp_mask * self.w, axis=0)

        self.sensitive = sensitive
        # all sensitive attributes are included by default
        if len(sensitive) == 0:
            self.sensitive = list(dataset.privileged.keys())

        # check for available sensitive attributes
        for attr in self.sensitive:
            if attr not in self.priviledged:
                raise ValueError(f"invalid sensitive value: \'{attr}\' is not in the dataset.")
    
    def sensitive_id(self):
        available = list(self.dataset.privileged.keys())
        return map(available.index, self.sensitive)

    def favorable_rate(self, privileged=None):
        mask = None
        result = {}
        if privileged is None:
            mask = np.ones_like(self.z) == 1
        elif privileged:
            mask = self.pri_mask
        else:
            mask = self.unp_mask

        if len(self.sensitive) == 1:
            i  = list(self.dataset.privileged.keys()).index(self.sensitive[0])
            y_z = self.y[mask[:, i]]
            w_z = self.w[mask[:, i]]
            total = int(np.sum(w_z))
            return np.sum(y_z * w_z) / total

        # go through sensitive attributes
        for (i, attr) in enumerate(self.sensitive):
            y_z = self.y[mask[:, i]]
            w_z = self.w[mask[:, i]]
            total = int(np.sum(w_z))
            result[attr] = np.sum(y_z * w_z) / total
        
        return result

    def favorable_diff(self):
        return dic_operation(self.favorable_rate(True), self.favorable_rate(False), lambda x, y: x - y)

    def favorable_ratio(self):
        return dic_operation(self.favorable_rate(False), self.favorable_rate(True), lambda x, y: (x / y) if y!=0 else 1 if x==y else x/1e-6)

    def consistency(self, n_neighbors=5):

        x = np.array(self.dataset.X)
        y = np.array(self.dataset.Y)
        return consistency(x, y, n_neighbors)
        



    

    