from fairness_datasets import FairnessDataset
from metrics.dataset_metric import DatasetMetrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from metrics.metric_utils import *

# def dic_operation(dic1, dic2, func):
#     result = {}
#     for key in dic1:
#         result[key] = func(dic1[key], dic2[key])
#     return result

# select out priviledged and unpriiviledged samples from dataset
class ModelMetrics(DatasetMetrics): # binary label and binary group only

    FAIRNESS_METRICS={
        # label-agnostic group fairness metrics
        "DI" : disparate_impact,
        "DP": demographic_parity,
        # single group evaluation metric(with out sensitive attribute z)
        "OMd": overall_misc,
        "OMr": overall_misc,
        "FPd": false_positive,
        "FPr": false_positive,
        "TPd": true_positive,
        "TPr": true_positive,
        "FNd": false_negtive,
        "FNr": false_negtive,
        "TNd": true_negtive,
        "TNr": true_negtive,
        "FOd": false_omission,
        "FOr": false_omission,
        "FDd": false_discovery,
        "FDr": false_discovery,
        # prediction fairness metrics
        "PE": predictive_equality,
        "EOD": equal_odds,
        "PP": predictive_parity,
    }

    BASE_METRICS={
        "FPR": false_positive,
        "TPR": true_positive,
        "FNR": false_negtive,
        "TNR": true_negtive,
        # "PR":,
        # "FR":
    }

    # sensitive attributes and label are given by dataset
    def __init__(self, dataset: FairnessDataset, classified_dataset: FairnessDataset, thresh=0.5, sensitive=[]):
        super().__init__(dataset=dataset, sensitive=sensitive)
        # self.dataset = dataset
        self.classifed_dataset = classified_dataset
        # self.priviledged = dataset.privileged
        # self.favorable = dataset.favorable
        self.thresh = thresh

        # extract vector of sensitive attribute and label
        # self.z, self.y = np.array(dataset.Z), np.array(dataset.Y)
        self.yh = np.array(classified_dataset.Y)
        # self.w = dataset.weights

        # mask for privileged an unprivileged
        # self.pri_mask = (self.z == 1)
        # self.unp_mask = (self.z != 1)

        # statistics of groups
        # self.total_num = np.sum(self.w)
        # self.pri_num = np.sum(self.pri_mask * self.w, axis=0)
        # self.unp_num = np.sum(self.unp_mask * self.w, axis=0)
        # if len(self.sensitive) > 1:
        #     self.sensitive = self.sensitive[:1] # only a single sensitive attribute supported

    def group_fairness_metrics(self, metrics="DP"):
        if metrics not in list(ModelMetrics.FAIRNESS_METRICS.keys()):
            raise ValueError(f"Metrices \'{metrics}\' not supported.")
        func = ModelMetrics.FAIRNESS_METRICS[metrics]
        if metrics.endswith('d'):
            return self.diff(func)
        elif metrics.endswith('r'):
            return self.ratio(func)
        else:
            return self.base_rate(func)

    def base_metrics(self, metrics="FPR", privileged=None):
        if metrics not in list(ModelMetrics.BASE_METRICS.keys()):
            raise ValueError(f"Metrices \'{metrics}\' not supported.")
        func = ModelMetrics.BASE_METRICS[metrics]
        return self.base_rate(func, privileged)


    def base_rate(self,  func, favorable=None):
        mask = None
        result = {}
        if favorable is None:
            mask = np.ones_like(self.z) == 1
        elif favorable:
            mask = self.pri_mask
        else:
            mask = self.unp_mask

        if len(self.sensitive) == 1:
            i  = list(self.dataset.privileged.keys()).index(self.sensitive[0])
            if func.__code__.co_argcount > 4:
                return func(self.yh, self.y, self.z[:, i], self.w, self.thresh)
            y_z = self.y[mask[:, i]]
            yh_z = self.yh[mask[:, i]]
            w_z = self.w[mask[:, i]]
            # z_z = self.z[mask[:, i]][:, i]
            return func(yh_z, y_z, w_z, self.thresh)
        
        # go through sensitive attributes
        for (i, attr) in enumerate(self.sensitive):
            if favorable is None: # for none single group metrics
                result[attr] = func(self.yh, self.y, self.z[:, i], self.w, self.thresh)
                continue
            y_z = self.y[mask[:, i]]
            yh_z = self.yh[mask[:, i]]
            w_z = self.w[mask[:, i]]
            # z_z = self.z[mask[:, i]][:, i]
            result[attr] = func(yh_z, y_z, w_z, self.thresh)
        
        return result

    def diff(self, func):
        return dic_operation(self.base_rate(func, True), self.base_rate(func, False), lambda x, y: x - y)

    def ratio(self, func):
        return dic_operation(self.base_rate(func, False), self.base_rate(func, True), lambda x, y: x / (y+1e-6))

    def consistency(self, n_neighbors=5):
        r"""Individual fairness metric from [1]_ that measures how similar the
        labels are for similar instances.

        .. math::
            1 - \frac{1}{n\cdot\text{n_neighbors}}\sum_{i=1}^n |\hat{y}_i -
            \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        Args:
            n_neighbors (int, optional): Number of neighbors for the knn
                computation.

        References:
            .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
                "Learning Fair Representations,"
                International Conference on Machine Learning, 2013.
        """

        X = np.array(self.dataset.X)
        num_samples = X.shape[0]
        y = self.yh

        # learn a KNN on the features
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)

        # compute consistency score
        consistency = 0.0
        for i in range(num_samples):
            consistency += np.abs(y[i] - np.mean(y[indices[i]])) * self.w[i]
        consistency = 1.0 - consistency/num_samples

        return consistency
        



    

    