import numpy as np
from sklearn.metrics import confusion_matrix

# model metrics

def get_confusion_components(y_true, y_pred, attr):
    """Calculate confusion matrix components for each demographic group using numpy operations"""
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true[attr==1], y_pred[attr==1])
    
    # compute tp, fp, tn, fn
    tp = np.diag(cm) # TP is the diagonal elements
    fp = cm.sum(axis=0) - tp # FP = sum columns - TP
    fn = cm.sum(axis=1) - tp # FN = sum rows - TP
    tn = cm.sum() - (fp + fn + tp) # TN = sum total - (FP+FN+TP)

    return tp, fp, tn, fn

def mean_precision(TP, FP, FN, TN):
    """Calculate mean precision (macro-average)"""
    try:
        precision = TP / (TP + FP)
        return np.nanmean(precision)
    except ZeroDivisionError:
        return 0

def mean_fpr(TP, FP, FN, TN):
    """Calculate mean false positive rate (FPR) (macro-average)"""
    try:
        fpr = FP / (FP + TN)
        return np.nanmean(fpr)
    except ZeroDivisionError:
        return 0

def mean_fnr(TP, FP, FN, TN):
    """Calculate mean false negative rate (FNR) (macro-average)"""
    try:
        fnr = FN / (TP + FN)
        return np.nanmean(fnr)
    except ZeroDivisionError:
        return 0

def mean_tnr(TP, FP, FN, TN):
    """Calculate mean true negative rate (TNR) (macro-average)"""
    try:
        tnr = TN / (TN + FP)
        return np.nanmean(tnr)
    except ZeroDivisionError:
        return 0

def mean_tpr(TP, FP, FN, TN):
    """Calculate mean true positive rate (TPR) (macro-average)"""
    try:
        tpr = TP / (TP + FN)
        return np.nanmean(tpr)
    except ZeroDivisionError:
        return 0

def mean_accuracy(TP, FP, FN, TN):
    """Calculate mean accuracy (macro-average)"""
    try:
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        return np.nanmean(accuracy)
    except ZeroDivisionError:
        return 0

def mean_recall(TP, FP, FN, TN):
    """Calculate mean recall (True Positive Rate) (macro-average)"""
    try:
        recall = TP / (TP + FN)
        return np.nanmean(recall)
    except ZeroDivisionError:
        return 0

def mean_specificity(TP, FP, FN, TN):
    """Calculate mean specificity (True Negative Rate) (macro-average)"""
    try:
        specificity = TN / (TN + FP)
        return np.nanmean(specificity)
    except ZeroDivisionError:
        return 0

def mean_f1_score(TP, FP, FN, TN):
    """Calculate mean F1 score (macro-average)"""
    precision = mean_precision(TP, FP, FN, TN)
    recall = mean_recall(TP, FP, FN, TN)
    try:
        f1_score = 2 * ((precision * recall) / (precision + recall))
        return np.nanmean(f1_score)
    except ZeroDivisionError:
        return 0

def mean_balanced_accuracy(TP, FP, FN, TN):
    """Calculate mean balanced accuracy (macro-average)"""
    recall = mean_recall(TP, FP, FN, TN)
    specificity = mean_specificity(TP, FP, FN, TN)
    return (recall + specificity) / 2



def calculate_demographic_parity(y_pred, y_true, z, metric_func, thd=0.5):
    """Calculate the difference between the metrics of two demographic groups"""
    np.seterr(divide='ignore', invalid='ignore')
    
    # Get the confusion matrix components for each group
    group1_cm = get_confusion_components(y_true, y_pred, z == 0)
    group2_cm = get_confusion_components(y_true, y_pred, z == 1)
    
    # Compute the metrics for each group
    metric_group1 = metric_func(*group1_cm)
    metric_group2 = metric_func(*group2_cm)
    
    # Return the absolute difference
    return abs(metric_group1 - metric_group2)


FAIRNESS_METRICS = {
    "mPre": mean_precision,
    "mFPR": mean_fpr,
    "mFNR": mean_fnr,
    "mTNR": mean_tnr,
    "mTPR": mean_tpr,
    "mAcc": mean_accuracy,
    "mF1": mean_f1_score,
    "mBA": mean_balanced_accuracy
}

METRICS_FULL_NAME = {
    "mPre": "Mean Precision Difference",
    "mFPR": "Mean False Positive Rate Difference",
    "mFNR": "Mean False Negative Rate Difference",
    "mTNR": "Mean True Negative Rate Difference",
    "mTPR": "Mean True Positive Rate Difference",
    "mAcc": "Mean Accuracy Difference",
    "mF1": "Mean F1 Score Difference",
    "mBA": "Mean Balanced Accuracy Difference"
}

# dataset metrics
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import math

def calculate_weighted_entropy(dist, weights):
    return entropy(dist.values, base=dist.shape[0], weights=weights)

def calculate_weighted_gini(dist, weights):
    # Weighted Gini coefficient using the ROC AUC scoring function
    return roc_auc_score(np.arange(len(dist)), dist.values, sample_weight=weights) * 2 - 1

def calculate_data_metrics(demographic_data, label_data, weights_data):
    # Create a DataFrame
    df = pd.DataFrame({
        'demographic': demographic_data,
        'label': label_data,
        'weights': weights_data
    })

    # Calculate distribution per class label, taking weights into account
    class_distribution = df.groupby('label').apply(lambda x: (x['demographic'].value_counts() * x['weights']).sum() / x['weights'].sum()).unstack().fillna(0)

    # Calculate weighted entropy and weighted Gini coefficient for each class
    entropy_values = class_distribution.apply(calculate_weighted_entropy, weights=df['weights'], axis=1)
    gini_values = class_distribution.apply(calculate_weighted_gini, weights=df['weights'], axis=1)

    # Calculate average weighted entropy and weighted Gini coefficient
    avg_entropy = entropy_values.mean()
    avg_gini = gini_values.mean()

    # Normalize
    num_demographics = len(df['demographic'].unique())
    normalized_entropy = avg_entropy / math.log(num_demographics)
    normalized_gini = (avg_gini + 1) / 2

    return normalized_entropy, normalized_gini

if __name__ == "__main__":
    # Assuming your data is in three numpy arrays
    demographic_data = np.array([...])  # replace with your demographic data
    label_data = np.array([...])  # replace with your label data
    weights_data = np.array([...])  # replace with your weights data

    normalized_entropy, normalized_gini = calculate_data_metrics(demographic_data, label_data, weights_data)

    print("Weighted Normalized Entropy:", normalized_entropy)
    print("Weighted Normalized Gini Coefficient:", normalized_gini)
