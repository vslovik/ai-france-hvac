from ml_evaluation.dm_lift import plot_lift_analysis, get_top_decile_lift
from ml_evaluation.evaluation import plot_roc_pr_ks, ks_interpretation
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def model_evaluation_report(df, model, features, target):
    """
    Complete model validation with all visualizations.
    """
    X = df[features]
    y_true = df[target].values

    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)

    print("=" * 80)
    print("COMPLETE MODEL VALIDATION PACKAGE")
    print("=" * 80)

    # 1. Discrimination Power Metrics
    print("\n📊 STEP 1: DISCRIMINATION POWER")
    print("-" * 40)

    roc_auc, pr_auc, ks_statistic, optimal_threshold = plot_roc_pr_ks(y_true, y_pred_proba)

    print(f"• ROC-AUC: {roc_auc:.4f}")
    print(f"• PR-AUC: {pr_auc:.4f}")
    print(f"• KS Statistic: {ks_statistic:.3f} ({ks_interpretation(ks_statistic)})")
    print(f"• Optimal Threshold: {optimal_threshold:.3f}")

    # 2. Business Impact (Lift)
    print("\n💼 STEP 2: BUSINESS IMPACT")
    print("-" * 40)

    decile_stats = plot_lift_analysis(y_true, y_pred_proba)
    top_decile_lift = get_top_decile_lift(y_true, y_pred_proba)

    print(f"• Top Decile Lift: {top_decile_lift:.2f}x")
    print(f"• Top 30% captures: {decile_stats.iloc[2]['cum_positives_pct']:.1f}% of positives")
    print(f"• Baseline response rate: {y_true.mean() * 100:.1f}%")
    print(f"• Top decile response: {decile_stats.iloc[0]['response_rate'] * 100:.1f}%")

    # 4. Confusion Matrix at Optimal Threshold
    print("\n📈 STEP 4: CONFUSION MATRIX")
    print("-" * 40)

    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.3f})', fontsize=14)
    plt.colorbar()

    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"At threshold {optimal_threshold:.3f}:")
    print(f"• Accuracy: {accuracy:.3f}")
    print(f"• Precision: {precision:.3f}")
    print(f"• Recall: {recall:.3f}")
    print(f"• F1-Score: {f1:.3f}")
    print(f"• True Positives: {tp}")
    print(f"• False Positives: {fp}")
    print(f"• True Negatives: {tn}")
    print(f"• False Negatives: {fn}")

    # 5. Final Summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\n✅ DISCRIMINATION POWER:")
    print(f"   KS Statistic: {ks_statistic:.3f} ({ks_interpretation(ks_statistic)})")

    print(f"\n✅ BUSINESS VALUE:")
    print(f"   Efficiency: {top_decile_lift:.2f}x better than random")
    print(f"   Coverage: Top 30% → {decile_stats.iloc[2]['cum_positives_pct']:.1f}% captured")

    print(f"\n✅ MODEL QUALITY:")
    print(f"   ROC-AUC: {roc_auc:.4f} (Excellent if >0.8)")
    print(f"   PR-AUC: {pr_auc:.4f} (Excellent if >0.7)")

    print(f"\n✅ OPERATIONAL METRICS:")
    print(f"   Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   Precision at threshold: {precision:.3f}")
    print(f"   Recall at threshold: {recall:.3f}")

    # Return all results
    return {
        'y_true': y_true,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ks_statistic': ks_statistic,
        'optimal_threshold': optimal_threshold,
        'top_decile_lift': top_decile_lift,
        'decile_stats': decile_stats,
        'confusion_matrix': cm,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }