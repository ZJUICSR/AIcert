from tabular.fairness_datasets import FairnessDataset
import numpy as np
from tabular.metrics.dataset_metric import DatasetMetrics
from tabular.metrics.model_metrics import ModelMetrics

class PostProcess:
    def __init__(self):
        pass

    def fit(self, dataset: FairnessDataset, dataset_pred:FairnessDataset):
        pass

    def transform(self, dataset: FairnessDataset, **kwargs):
        pass

class CalibratedEqOdds(PostProcess):
    def __init__(self, cost_constraint='fnr', sensitive=[]):
        self.cost_constraint = cost_constraint
        self.sensitive = sensitive

        if self.cost_constraint == 'fnr':
            self.fn_rate = 1
            self.fp_rate = 0
        elif self.cost_constraint == 'fpr':
            self.fn_rate = 0
            self.fp_rate = 1
        elif self.cost_constraint == 'weighted':
            self.fn_rate = 1
            self.fp_rate = 1

        self.base_rate_priv = 0.0
        self.base_rate_unpriv = 0.0
        if len(self.sensitive) > 1:
            raise ValueError('only a single sensitive attribue are supported')

    def check_sens(self, dataset):
        available = list(dataset.privileged.keys())
        if len(self.sensitive) == 0:
            self.sensitive = [available[0]]
        if self.sensitive[0] not in available:
            raise ValueError(f"invalid sensitive value: \'{self.sensitive[0]}\' is not in the dataset.")
        idx = available.index(self.sensitive[0])
        return idx

    def fit(self, dataset: FairnessDataset, dataset_pred:FairnessDataset):
        idx = self.check_sens(dataset=dataset)
        mm = ModelMetrics(dataset, dataset_pred, sensitive=self.sensitive)
        self.base_rate_priv = mm.favorable_rate(True)
        self.base_rate_unpriv = mm.favorable_rate(False)
        
        pri_mask = mm.pri_mask[:, idx]
        unp_mask = mm.unp_mask[:, idx]

        # Create a dataset with "trivial" predictions
        dataset_trivial = dataset_pred.copy(deepcopy=True)
        dataset_trivial.Y[pri_mask] = mm.favorable_rate(privileged=True)
        dataset_trivial.Y[unp_mask] = mm.favorable_rate(privileged=False)

        mm_triv = ModelMetrics(dataset, dataset_trivial, sensitive=self.sensitive)

        if self.fn_rate == 0:
            priv_cost = mm.general_base_metrics(metrics="FPR",privileged=True)
            unpriv_cost = mm.general_base_metrics(metrics="FPR",privileged=False)
            priv_trivial_cost = mm_triv.general_base_metrics(metrics="FPR",privileged=True)
            unpriv_trivial_cost = mm_triv.general_base_metrics(metrics="FPR",privileged=False)

        elif self.fp_rate == 0:
            priv_cost = mm.general_base_metrics(metrics="FNR", privileged=True)
            unpriv_cost = mm.general_base_metrics(metrics="FNR", privileged=False)
            priv_trivial_cost = mm_triv.general_base_metrics(metrics="FNR",privileged=True)
            unpriv_trivial_cost = mm_triv.general_base_metrics(metrics="FNR",privileged=False)

        else:
            priv_cost = self.weighted_cost(self.fp_rate, self.fn_rate, mm, privileged=True)
            unpriv_cost = self.weighted_cost(self.fp_rate, self.fn_rate, mm, privileged=False)
            priv_trivial_cost = self.weighted_cost(self.fp_rate, self.fn_rate, mm_triv, privileged=True)
            unpriv_trivial_cost = self.weighted_cost(self.fp_rate, self.fn_rate, mm_triv, privileged=False)

        unpriv_costs_more = unpriv_cost > priv_cost
        self.priv_mix_rate = (unpriv_cost - priv_cost) / (priv_trivial_cost - priv_cost) if unpriv_costs_more else 0
        self.unpriv_mix_rate = 0 if unpriv_costs_more else (priv_cost - unpriv_cost) / (unpriv_trivial_cost - unpriv_cost + 1e-6)

        return self

    def transform(self, dataset: FairnessDataset, threshold=0.5):
        """Perturb the predicted scores to obtain new labels that satisfy
        equalized odds constraints, while preserving calibration.
        Args:
            dataset (BinaryLabelDataset): Dataset containing `scores` that needs
                to be transformed.
            threshold (float): Threshold for converting `scores` to `labels`.
                Values greater than or equal to this threshold are predicted to
                be the `favorable_label`. Default is 0.5.
        Returns:
            dataset (BinaryLabelDataset): transformed dataset.
        """
        idx = self.check_sens(dataset=dataset)
        dm = DatasetMetrics(dataset=dataset)
        pri_mask = dm.pri_mask[:, idx]
        unp_mask = dm.unp_mask[:, idx]

        unpriv_indices = (np.random.random(sum(unp_mask))
                       <= self.unpriv_mix_rate)
        unpriv_new_pred = dataset.Y[unp_mask].copy()
        unpriv_new_pred[unpriv_indices] = self.base_rate_unpriv

        priv_indices = (np.random.random(sum(pri_mask))
                     <= self.priv_mix_rate)
        priv_new_pred = dataset.Y[pri_mask].copy()
        priv_new_pred[priv_indices] = self.base_rate_priv

        dataset_new = dataset.copy(deepcopy=True)

        dataset_new.Y = np.zeros_like(dataset.Y, dtype=np.float64)
        dataset_new.Y[pri_mask] = priv_new_pred
        dataset_new.Y[unp_mask] = unpriv_new_pred

        # Create labels from scores using a default threshold
        dataset_new.Y = np.where(dataset_new.Y >= threshold, 1, 0)
        return dataset_new


    def weighted_cost(self, fpr, fnr, mm: ModelMetrics, privileged):
        norm_const = float(fpr + fnr) if (fpr != 0 and fnr != 0) else 1

        return ((fpr / norm_const
                * mm.general_base_metrics(metrics="FPR",privileged=privileged)
                * (1 - mm.favorable_rate(privileged=privileged))) +
            (fnr / norm_const
                * mm.general_base_metrics(metrics="FNR",privileged=privileged)
                * mm.favorable_rate(privileged=privileged)))


class RejectOptionClassification(PostProcess):
    def __init__(self, sensitive=[], low_class_thresh=0.01, high_class_thresh=0.99,
                num_class_thresh=100, num_ROC_margin=50,
                metric_name="SPd",
                metric_ub=1, metric_lb=-1):

        allowed_metrics = ["SPd",
                           "AOd",
                           "EOd"]
        self.sensitive = sensitive
        self.low_class_thresh = low_class_thresh
        self.high_class_thresh = high_class_thresh
        self.num_class_thresh = num_class_thresh
        self.num_ROC_margin = num_ROC_margin
        self.metric_name = metric_name
        self.metric_ub = metric_ub
        self.metric_lb = metric_lb

        self.classification_threshold = None
        self.ROC_margin = None

        if ((self.low_class_thresh < 0.0) or (self.low_class_thresh > 1.0) or\
            (self.high_class_thresh < 0.0) or (self.high_class_thresh > 1.0) or\
            (self.low_class_thresh >= self.high_class_thresh) or\
            (self.num_class_thresh < 1) or (self.num_ROC_margin < 1)):

            raise ValueError("Input parameter values out of bounds")

        if metric_name not in allowed_metrics:
            raise ValueError("metric name not in the list of allowed metrics")
        
        if len(self.sensitive) > 1:
            raise ValueError('only a single sensitive attribue are supported')

    def fit(self, dataset, dataset_pred):
        fair_metric_arr = np.zeros(self.num_class_thresh*self.num_ROC_margin)
        balanced_acc_arr = np.zeros_like(fair_metric_arr)
        ROC_margin_arr = np.zeros_like(fair_metric_arr)
        class_thresh_arr = np.zeros_like(fair_metric_arr)

        dataset_pred.Y = (dataset_pred.Y-np.min(dataset_pred.Y))/(np.max(dataset_pred.Y)-np.min(dataset_pred.Y))

        cnt = 0
        # Iterate through class thresholds
        for class_thresh in np.linspace(self.low_class_thresh,
                                        self.high_class_thresh,
                                        self.num_class_thresh):

            self.classification_threshold = class_thresh
            if class_thresh <= 0.5:
                low_ROC_margin = 0.0
                high_ROC_margin = class_thresh
            else:
                low_ROC_margin = 0.0
                high_ROC_margin = (1.0-class_thresh)

            # Iterate through ROC margins
            for ROC_margin in np.linspace(
                                low_ROC_margin,
                                high_ROC_margin,
                                self.num_ROC_margin):
                self.ROC_margin = ROC_margin

                # Predict using the current threshold and margin
                dataset_transf_pred = self.transform(dataset_pred, deepcopy=True)

                dataset_transf_metric_pred = DatasetMetrics(dataset_transf_pred, sensitive=self.sensitive)
                classified_transf_metric = ModelMetrics(dataset, dataset_transf_pred, sensitive=self.sensitive)

                ROC_margin_arr[cnt] = self.ROC_margin
                class_thresh_arr[cnt] = self.classification_threshold

                # Balanced accuracy and fairness metric computations
                balanced_acc_arr[cnt] = 0.5*(classified_transf_metric.base_metrics("TPR")\
                                       +classified_transf_metric.base_metrics("TNR"))
                if self.metric_name == "SPd":
                    fair_metric_arr[cnt] = dataset_transf_metric_pred.favorable_diff()
                elif self.metric_name == "AOd":
                    # Average of difference in FPR and TPR for unprivileged and privileged groups
                    fair_metric_arr[cnt] = 0.5 * (classified_transf_metric.group_fairness_metrics(metrics="TPd") + classified_transf_metric.group_fairness_metrics(metrics="FPd"))
                elif self.metric_name == "EOd":
                    fair_metric_arr[cnt] = classified_transf_metric.group_fairness_metrics(metrics="TPd")

                cnt += 1

        rel_inds = np.logical_and(fair_metric_arr >= self.metric_lb,
                                  fair_metric_arr <= self.metric_ub)
        if any(rel_inds):
            best_ind = np.where(balanced_acc_arr[rel_inds]
                                == np.max(balanced_acc_arr[rel_inds]))[0][0]
        else:
            print("Warning: Unable to satisy fairness constraints")
            rel_inds = np.ones(len(fair_metric_arr), dtype=bool)
            best_ind = np.where(fair_metric_arr[rel_inds]
                                == np.min(fair_metric_arr[rel_inds]))[0][0]

        self.ROC_margin = ROC_margin_arr[rel_inds][best_ind]
        self.classification_threshold = class_thresh_arr[rel_inds][best_ind]

        return self

    def check_sens(self, dataset):
        available = list(dataset.privileged.keys())
        if len(self.sensitive) == 0:
            self.sensitive = [available[0]]
        if self.sensitive[0] not in available:
            raise ValueError(f"invalid sensitive value: \'{self.sensitive[0]}\' is not in the dataset.")
        idx = available.index(self.sensitive[0])
        return idx

    def transform(self, dataset:FairnessDataset, deepcopy=False):
        """Obtain fair predictions using the ROC method.
        Args:
            dataset (BinaryLabelDataset): Dataset containing scores that will
                be used to compute predicted labels.
        Returns:
            dataset_pred (BinaryLabelDataset): Output dataset with potentially
            fair predictions obtain using the ROC method.
        """
        dataset_new = dataset.copy(deepcopy=deepcopy)

        fav_pred_inds = (dataset.Y > self.classification_threshold)
        unfav_pred_inds = ~fav_pred_inds

        y_pred = np.zeros(dataset.Y.shape)
        y_pred[fav_pred_inds] = 1
        y_pred[unfav_pred_inds] = 0

        # Indices of critical region around the classification boundary
        crit_region_inds = np.logical_and(
                dataset.Y <= self.classification_threshold+self.ROC_margin,
                dataset.Y > self.classification_threshold-self.ROC_margin)

        # # Indices of privileged and unprivileged groups
        # cond_priv = utils.compute_boolean_conditioning_vector(
        #                 dataset.protected_attributes,
        #                 dataset.protected_attribute_names,
        #                 self.privileged_groups)
        # cond_unpriv = utils.compute_boolean_conditioning_vector(
        #                 dataset.protected_attributes,
        #                 dataset.protected_attribute_names,
        #                 self.unprivileged_groups)

        dm = DatasetMetrics(dataset)
        idx = self.check_sens(dataset=dataset)
        pri_mask = dm.pri_mask[:, idx]
        unp_mask = dm.unp_mask[:, idx]
        # New, fairer labels
        dataset_new.Y = y_pred
        dataset_new.Y[np.logical_and(crit_region_inds,
                            pri_mask.reshape(-1,1))] = 0
        dataset_new.Y[np.logical_and(crit_region_inds,
                            unp_mask.reshape(-1,1))] = 1

        return dataset_new


