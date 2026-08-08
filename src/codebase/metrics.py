import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)


def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs


def compute_accuracy(gt, pred):
    return (((pred == gt).sum()) / gt.size(0)).item() * 100


def compute_auprc(gt, pred):
    return average_precision_score(gt, pred)


def compute_accuracy_np_array(gt, pred):
    return np.mean(gt == pred)


def pr_auc(gt, pred, get_all=False):
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score


# https://www.kaggle.com/code/sohier/probabilistic-f-score
def pfbeta(gt, pred, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gt)):
        prediction = min(max(pred[idx], 0), 1)
        if (gt[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def all_classification_metrics(gt, pred, target_fpr=0.15):
    # Apply threshold to convert probabilities to class predictions
    metrics = {
        "AUROC": roc_auc_score(gt, pred),
        "AUPRC": average_precision_score(gt, pred)
    }

    return metrics


def compute_opt_thres(y_true, y_pred, target_fpr=0.15):
    """Pick the operating threshold whose false-positive rate is closest to target_fpr.

    Args:
        y_true: array-like of binary ground-truth labels (0/1).
        y_pred: array-like of predicted probabilities/scores for the positive class.
        target_fpr: desired false-positive rate of the operating point.

    Returns:
        float threshold, to be applied as `y_pred >= threshold`. Falls back to 0.5
        when a ROC curve cannot be computed (e.g., y_true contains a single class).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.unique(y_true).size < 2:
        print("[compute_opt_thres] y_true contains a single class; falling back to threshold=0.5")
        return 0.5
    fpr, _, thresholds = roc_curve(y_true, y_pred)
    # roc_curve prepends an infinite threshold; keep finite ones only
    finite = np.isfinite(thresholds)
    fpr, thresholds = fpr[finite], thresholds[finite]
    if thresholds.size == 0:
        return 0.5
    dist = np.abs(fpr - target_fpr)
    # among ROC points equally close to target_fpr, take the last (highest TPR)
    idx = np.where(dist == dist.min())[0][-1]
    return float(thresholds[idx])


def auroc(gt, pred):
    return roc_auc_score(gt, pred)


def pfbeta_binarized(gt, pred):
    positives = pred[gt == 1]
    scores = []
    for th in positives:
        binarized = (pred >= th).astype('int')
        score = pfbeta(gt, binarized, 1)
        scores.append(score)

    return np.max(scores)
