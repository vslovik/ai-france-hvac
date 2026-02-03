import pandas as pd


def find_optimal_threshold(y_true, y_pred_proba, cost_fp=1, cost_fn=2):
    """
    Find optimal threshold considering business costs
    cost_fp: Cost of false positive (wasted sales time)
    cost_fn: Cost of false negative (missed conversion)
    """
    from sklearn.metrics import confusion_matrix

    thresholds = np.arange(0.30, 0.55, 0.01)
    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate total cost
        total_cost = (fp * cost_fp) + (fn * cost_fn)

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fn_rate = fn / (fn + tp)  # Miss rate among actual positives
        fp_rate = fp / (fp + tn)  # False alarm rate among actual negatives

        results.append({
            'threshold': threshold,
            'total_cost': total_cost,
            'accuracy': accuracy,
            'fn_rate': fn_rate,
            'fp_rate': fp_rate,
            'fn_count': fn,
            'fp_count': fp,
            'tp_count': tp,
            'tn_count': tn
        })

    results_df = pd.DataFrame(results)

    # Find minimum cost threshold
    optimal_row = results_df.loc[results_df['total_cost'].idxmin()]

    print("Optimal threshold analysis:")
    print(f"Current threshold: 0.487")
    print(f"Optimal threshold: {optimal_row['threshold']:.3f}")
    print(
        f"Expected reduction in total cost: {results_df.loc[results_df['threshold'] == 0.487, 'total_cost'].values[0] - optimal_row['total_cost']:.0f}")

    # Show trade-off
    print(f"\nTrade-off at optimal threshold {optimal_row['threshold']:.3f}:")
    print(f"  • False Negatives: {optimal_row['fn_count']} (missed opportunities)")
    print(f"  • False Positives: {optimal_row['fp_count']} (wasted effort)")
    print(f"  • Accuracy: {optimal_row['accuracy']:.3f}")

    return results_df, optimal_row
