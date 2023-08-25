from image.metrics.multilabel_metrics import calculate_demographic_parity as cal_ml_metrics
from image.metrics.singlelabel_metrics import calculate_demographic_parity as cal_sl_metrics

from image.metrics.multilabel_metrics import FAIRNESS_METRICS as ml_metrics
from image.metrics.singlelabel_metrics import FAIRNESS_METRICS as sl_metrics

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