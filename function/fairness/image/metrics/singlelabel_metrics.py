import numpy as np
from sklearn.metrics import confusion_matrix

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


