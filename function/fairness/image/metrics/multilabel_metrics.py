from sklearn.metrics import multilabel_confusion_matrix
import numpy as np


def get_confusion_components(y_true, y_pred, attr):
    """Calculate confusion matrix components for each demographic group using numpy operations"""
    
    # Compute sample wise multi-label confusion matrix
    mcm = multilabel_confusion_matrix(y_true[attr==1], y_pred[attr==1], samplewise=True)
    
    # compute tp, fp, tn, fn
    sum_mcm = np.sum(mcm, axis=0)
    
    # Compute rates
    tp = sum_mcm[1, 1] # TP
    fn = sum_mcm[1, 0] # FN
    tn = sum_mcm[0, 0] # TN
    fp = sum_mcm[0, 1] # FP

    return tp, fp, tn, fn


def mean_precision(TP, FP, FN, TN):
    """Calculate mean precision"""
    try:
        return TP / (TP + FP)
    except ZeroDivisionError:
        return 0

def mean_fpr(TP, FP, FN, TN):
    """Calculate mean false positive rate (FPR)"""
    try:
        return FP / (FP + TN)
    except ZeroDivisionError:
        return 0

def mean_fnr(TP, FP, FN, TN):
    """Calculate mean false negative rate (FNR)"""
    try:
        return FN / (TP + FN)
    except ZeroDivisionError:
        return 0

def mean_tnr(TP, FP, FN, TN):
    """Calculate mean true negative rate (TNR)"""
    try:
        return TN / (TN + FP)
    except ZeroDivisionError:
        return 0

def mean_tpr(TP, FP, FN, TN):
    """Calculate mean true positive rate (TPR)"""
    try:
        return TP / (TP + FN)
    except ZeroDivisionError:
        return 0


def mean_accuracy(TP, FP, FN, TN):
    """Calculate mean accuracy"""
    try:
        return (TP + TN) / (TP + FP + FN + TN)
    except ZeroDivisionError:
        return 0

def mean_recall(TP, FP, FN, TN):
    """Calculate mean recall (True Positive Rate)"""
    try:
        return TP / (TP + FN)
    except ZeroDivisionError:
        return 0

def mean_specificity(TP, FP, FN, TN):
    """Calculate mean specificity (True Negative Rate)"""
    try:
        return TN / (TN + FP)
    except ZeroDivisionError:
        return 0

def mean_f1_score(TP, FP, FN, TN):
    """Calculate mean F1 score"""
    precision = mean_precision(TP, FP, FN, TN)
    recall = mean_recall(TP, FP, FN, TN)
    try:
        return 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        return 0

def mean_balanced_accuracy(TP, FP, FN, TN):
    """Calculate mean balanced accuracy"""
    recall = mean_recall(TP, FP, FN, TN)
    specificity = mean_specificity(TP, FP, FN, TN)
    return (recall + specificity) / 2


def calculate_demographic_parity(y_pred, y_true, z, metric_func, thd=0.5):
    """Calculate the difference between the metrics of two demographic groups"""
    
    # threshold prediction
    y_pred = (y_pred > thd).astype(np.float32)
    
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
    "mRec": mean_recall,
    "mSpec": mean_specificity,
    "mF1": mean_f1_score,
    "mBA": mean_balanced_accuracy
}

METRICS_FULL_NAME = {
    "mPre": "Mean Precision",
    "mFPR": "Mean False Positive Rate",
    "mFNR": "Mean False Negative Rate",
    "mTNR": "Mean True Negative Rate",
    "mTPR": "Mean True Positive Rate",
    "mAcc": "Mean Accuracy",
    "mRec": "Mean Recall",
    "mSpec": "Mean Specificity",
    "mF1": "Mean F1 Score",
    "mBA": "Mean Balanced Accuracy"
}


