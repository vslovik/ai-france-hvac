import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import math


def ks_interpretation(ks):
    if ks >= 0.6:
        return "OUTSTANDING discrimination"
    elif ks >= 0.5:
        return "EXCELLENT discrimination"
    elif ks >= 0.4:
        return "GOOD discrimination"
    elif ks >= 0.3:
        return "MODERATE discrimination"
    else:
        return "WEAK discrimination"


def plot_roc_pr_ks(y_true, y_pred_proba):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    axes[0].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve - FIXED
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    # ✅ CRITICAL FIX: Ensure recall is non-decreasing
    # Sometimes sklearn returns points in weird order
    pr_pairs = np.column_stack([recall, precision])
    pr_pairs = pr_pairs[np.argsort(pr_pairs[:, 0])]  # Sort by recall
    recall, precision = pr_pairs[:, 0], pr_pairs[:, 1]

    pr_auc = auc(recall, precision)
    baseline = np.sum(y_true) / len(y_true)

    axes[1].plot(recall, precision, color='blue', lw=2,
                 label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[1].axhline(y=baseline, color='red', linestyle='--',
                    label=f'Baseline = {baseline:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    # 3. KS Statistic Plot
    ks_statistic = np.max(tpr - fpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    axes[2].plot(thresholds, tpr, 'b-', label='True Positive Rate', lw=2)
    axes[2].plot(thresholds, fpr, 'r-', label='False Positive Rate', lw=2)
    axes[2].fill_between(thresholds, fpr, tpr, alpha=0.3, color='gray')
    axes[2].axvline(x=optimal_threshold, color='green', linestyle='--',
                    label=f'Optimal = {optimal_threshold:.3f}')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Rate')
    axes[2].set_title(f'KS Statistic = {ks_statistic:.3f}')
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Model Discrimination Power', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return roc_auc, pr_auc, ks_statistic, optimal_threshold


def calculate_roc_auc(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc
    }


def calculate_pr_auc(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'auc': pr_auc
    }


def calculate_ks_statistic(y_true, y_pred_proba):
    df = pd.DataFrame({'true': y_true, 'prob': y_pred_proba})
    df = df.sort_values('prob', ascending=False)

    df['cum_pos'] = df['true'].cumsum() / df['true'].sum()
    df['cum_neg'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()

    ks = (df['cum_pos'] - df['cum_neg']).abs().max()
    ks_threshold = df.loc[(df['cum_pos'] - df['cum_neg']).abs().idxmax(), 'prob']

    return {
        'ks_statistic': ks,
        'ks_threshold': ks_threshold,
        'cum_pos': df['cum_pos'].values,
        'cum_neg': df['cum_neg'].values
    }


def calculate_confusion_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def calculate_lift_table(y_true, y_pred_proba, n_deciles=10):
    df = pd.DataFrame({'true': y_true, 'prob': y_pred_proba})
    df['decile'] = pd.qcut(df['prob'], q=n_deciles, labels=False) + 1

    lift_table = df.groupby('decile').agg({
        'prob': 'mean',
        'true': ['count', 'sum', 'mean']
    })
    lift_table.columns = ['avg_prob', 'count', 'positives', 'conversion_rate']

    overall_rate = df['true'].mean()
    lift_table['lift'] = lift_table['conversion_rate'] / overall_rate
    lift_table['cumulative_positives'] = lift_table['positives'].cumsum()

    return lift_table


def calculate_cohens_d_continuous(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    return {
        'cohens_d': cohens_d,
        'mean1': mean1, 'mean2': mean2,
        'std1': std1, 'std2': std2,
        'pooled_std': pooled_std
    }


def calculate_cohens_h_proportions(p1, p2, n1=None, n2=None):
    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    result = {'cohens_h': h, 'p1': p1, 'p2': p2}

    # Add confidence interval if sample sizes provided
    if n1 and n2:
        se = math.sqrt((1 / n1) + (1 / n2))
        result['se'] = se
        result['ci_lower'] = h - 1.96 * se
        result['ci_upper'] = h + 1.96 * se

    return result