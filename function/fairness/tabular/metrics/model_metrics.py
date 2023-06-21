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
METRICS_FULL_NAME = {
        # label-agnostic group fairness metrics
        "DI" : 'Dsiaprate Impact',
        "DP": 'Demographic Parity',
        # single group evaluation metric(with out sensitive attribute z)
        "OMd": 'Overall Misclassification Difference',
        "OMr": 'Overall Misclassification Ratio',
        "FPd": 'False Positive Difference',
        "FPr": 'False Positive Ratio',
        "FPn": "False Positive Number",
        "TPd": 'True Positive Difference',
        "TPr": 'True Positive Ratio',
        "TPn": 'True Positive Number',
        "FNd": 'False Negative Difference',
        "FNr": 'False Negative Ratio',
        "FNn": 'False Negative Number',
        "TNd": 'True Negative Difference',
        "TNr": 'True Negative Ratio',
        "TNn": 'True Negative Number',
        "FOd": 'False Omission Difference',
        "FOr": 'False Omission Ratio',
        "FDd": 'False Discovery Difference',
        "FDr": 'False Discovery Ratio',
        "PRd": "Precision Difference",
        "F1d": "F1 Score Difference",
        # prediction fairness metrics
        "PE": 'Predictive Equality',
        "EOP": 'Equal Opportunity',
        "EOD": 'Equal Odds',
        "PP": 'Predictive Parity',
    }

class ModelMetrics(DatasetMetrics): # binary label and binary group only

    FAIRNESS_METRICS={
        # label-agnostic group fairness metrics
        "DI" : disparate_impact,
        "DP": demographic_parity,
        "DP_norm": normalized_DP,

        # single group evaluation metric(with out sensitive attribute z)
        "OMd": overall_misc,
        "OMr": overall_misc,
        "FPn": false_positive_num,
        "FPd": false_positive,
        "FPr": false_positive,
        "TPn": true_positive_num,
        "TPd": true_positive,
        "TPr": true_positive,
        "FNn": false_negtive_num,
        "FNd": false_negtive,
        "FNr": false_negtive,
        "TNn": true_negtive_num,
        "TNd": true_negtive,
        "TNr": true_negtive,
        "FOd": false_omission,
        "FOr": false_omission,
        "FDd": false_discovery,
        "FDr": false_discovery,
        "PRd": precision,
        "F1d": F1_score,

        # prediction fairness metrics
        "PE": predictive_equality,
        "EOP": equal_opportunity,
        "EOD": equal_odds,
        "PP": predictive_parity,
        "TEE": treatment_equality,
        "PED": predictive_difference,
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
    def __init__(self, dataset: FairnessDataset, classified_dataset: FairnessDataset, sensitive=[]):
        super().__init__(dataset=dataset, sensitive=sensitive)
        # self.dataset = dataset
        self.classifed_dataset = classified_dataset
        # self.priviledged = dataset.privileged
        # self.favorable = dataset.favorable

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

    def group_fairness_metrics(self, metrics="DP", thd=0.5):
        if metrics not in list(ModelMetrics.FAIRNESS_METRICS.keys()):
            raise ValueError(f"Metrices \'{metrics}\' not supported.")
        func = ModelMetrics.FAIRNESS_METRICS[metrics]
        if metrics.endswith('d') or metrics.endswith('n'):
            return self.diff(func, thresh=thd)
        elif metrics.endswith('r'):
            return self.ratio(func)
        else:
            return self.base_rate(func, thresh=thd)

    def general_group_fairness_metrics(self, metrics="DP"):
        # the metrics are generalized when threshold is set to None
        return self.group_fairness_metrics(metrics=metrics, thd=None)

    def base_metrics(self, metrics="FPR", privileged=None, thd=0.5):
        if metrics not in list(ModelMetrics.BASE_METRICS.keys()):
            raise ValueError(f"Metrices \'{metrics}\' not supported.")
        func = ModelMetrics.BASE_METRICS[metrics]
        return self.base_rate(func, privileged, thresh=thd)

    def general_base_metrics(self, metrics="FPR", privileged=None):
        # the metrics are generalized when threshold is set to None
        return self.base_metrics(metrics=metrics, privileged=privileged, thd=None)


    def base_rate(self,  func, favorable=None, thresh=0.5):
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
                return func(self.yh, self.y, self.z[:, i], self.w, thresh)
            y_z = self.y[mask[:, i]]
            yh_z = self.yh[mask[:, i]]
            w_z = self.w[mask[:, i]]
            # z_z = self.z[mask[:, i]][:, i]
            return func(yh_z, y_z, w_z, thresh)
        
        # go through sensitive attributes
        for (i, attr) in enumerate(self.sensitive):
            if favorable is None: # for none single group metrics
                result[attr] = func(self.yh, self.y, self.z[:, i], self.w, thresh)
                continue
            y_z = self.y[mask[:, i]]
            yh_z = self.yh[mask[:, i]]
            w_z = self.w[mask[:, i]]
            # z_z = self.z[mask[:, i]][:, i]
            result[attr] = func(yh_z, y_z, w_z, thresh)
        
        return result

    def diff(self, func, **kwargs):
        return dic_operation(self.base_rate(func, True, **kwargs), self.base_rate(func, False, **kwargs), lambda x, y: abs(x - y))

    def ratio(self, func, **kwargs):
        return dic_operation(self.base_rate(func, False, **kwargs), self.base_rate(func, True, **kwargs), lambda x, y: x / (y+1e-6))

    def consistency(self, n_neighbors=5):

        x = np.array(self.dataset.X)
        y = self.yh
        return consistency(x, y)