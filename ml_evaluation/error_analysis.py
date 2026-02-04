import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def comprehensive_error_analysis(X_test, y_test, y_pred, y_pred_proba=None,
                                 original_df=None, customer_id_col='numero_compte'):
    """
    Comprehensive error analysis after model retraining
    """
    print("=" * 80)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 80)

    results = {}

    # ========== 1. BASIC ERROR METRICS ==========
    print("\n📊 1. BASIC ERROR METRICS")
    print("-" * 60)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:>6} (Correctly rejected)")
    print(f"  False Positives: {fp:>6} (Predicted YES, actual NO) ← WASTED EFFORT")
    print(f"  False Negatives: {fn:>6} (Predicted NO, actual YES) ← MISSED OPPORTUNITIES")
    print(f"  True Positives:  {tp:>6} (Correctly predicted)")

    # Error rates
    total = len(y_test)
    fp_rate = fp / total * 100
    fn_rate = fn / total * 100
    error_rate = (fp + fn) / total * 100

    print(f"\nError Analysis:")
    print(f"  • Total errors: {fp + fn:,} ({error_rate:.1f}% of predictions)")
    print(f"  • False Positive rate: {fp_rate:.1f}% (wasted sales effort)")
    print(f"  • False Negative rate: {fn_rate:.1f}% (missed conversions)")

    results['confusion_matrix'] = cm
    results['error_rates'] = {'fp_rate': fp_rate, 'fn_rate': fn_rate, 'total_error': error_rate}

    # ========== 2. CONFIDENCE ANALYSIS ==========
    if y_pred_proba is not None:
        print(f"\n📈 2. PREDICTION CONFIDENCE ANALYSIS")
        print("-" * 60)

        # Create confidence bins
        confidence_bins = pd.cut(y_pred_proba, bins=[0, 0.3, 0.4, 0.6, 0.7, 1.0],
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Analyze accuracy by confidence level
        confidence_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba,
            'confidence_level': confidence_bins
        })

        accuracy_by_confidence = confidence_df.groupby('confidence_level').apply(
            lambda x: (x['actual'] == x['predicted']).mean() * 100
        )

        print("Accuracy by Confidence Level:")
        for level, accuracy in accuracy_by_confidence.items():
            count = (confidence_df['confidence_level'] == level).sum()
            print(f"  • {level:10} confidence: {accuracy:.1f}% accurate ({count:,} predictions)")

        # Identify low-confidence predictions (need human review)
        low_confidence_mask = (y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)
        low_confidence_count = low_confidence_mask.sum()

        print(f"\n⚠️  Low Confidence Predictions (40-60%):")
        print(f"  • Count: {low_confidence_count:,} ({low_confidence_count / len(y_test) * 100:.1f}% of predictions)")
        print(f"  • These need human review or better features")

        results['confidence_analysis'] = confidence_df
        results['low_confidence_count'] = low_confidence_count

    # ========== 3. FEATURE IMPORTANCE IN ERRORS ==========
    print(f"\n🔍 3. FEATURE ANALYSIS OF ERRORS")
    print("-" * 60)

    if isinstance(X_test, pd.DataFrame):
        # Combine predictions with features
        error_df = X_test.copy()
        error_df['actual'] = y_test.values if hasattr(y_test, 'values') else y_test
        error_df['predicted'] = y_pred
        error_df['is_error'] = (y_test != y_pred).astype(int)
        error_df['error_type'] = np.select(
            [
                (y_test == 0) & (y_pred == 1),  # False Positive
                (y_test == 1) & (y_pred == 0),  # False Negative
                (y_test == y_pred) & (y_test == 1),  # True Positive
                (y_test == y_pred) & (y_test == 0)  # True Negative
            ],
            ['FP', 'FN', 'TP', 'TN'],
            default='Unknown'
        )

        # Analyze feature distributions for each error type
        numeric_features = error_df.select_dtypes(include=[np.number]).columns.tolist()

        print("Feature Value Differences by Error Type (mean values):")

        # Select key features based on your analysis
        key_features = [
            'total_quotes', 'avg_days_between_quotes', 'engagement_density',
            'solution_complexity_score', 'equipment_variety_count',
            'brand_loyalty_index', 'process_customer_fit', 'consideration_depth_score'
        ]

        # Filter to features that exist
        available_features = [f for f in key_features if f in error_df.columns]

        if available_features:
            error_stats = error_df.groupby('error_type')[available_features].mean()
            print(error_stats.round(3).to_string())

            results['error_feature_stats'] = error_stats

            # Identify features with biggest differences
            print(f"\n📊 Key Feature Differences:")

            for feature in available_features[:5]:  # Top 5 features
                fp_mean = error_stats.loc['FP', feature] if 'FP' in error_stats.index else np.nan
                fn_mean = error_stats.loc['FN', feature] if 'FN' in error_stats.index else np.nan
                tp_mean = error_stats.loc['TP', feature] if 'TP' in error_stats.index else np.nan

                if not np.isnan(fp_mean) and not np.isnan(tp_mean):
                    fp_vs_tp_diff = fp_mean - tp_mean
                    print(f"  • {feature:30}: FP are {fp_vs_tp_diff:+.3f} vs TP")

                if not np.isnan(fn_mean) and not np.isnan(tp_mean):
                    fn_vs_tp_diff = fn_mean - tp_mean
                    print(f"  • {feature:30}: FN are {fn_vs_tp_diff:+.3f} vs TP")

    # ========== 4. ENGAGEMENT-BASED ERROR ANALYSIS ==========
    print(f"\n🎯 4. ERROR ANALYSIS BY ENGAGEMENT LEVEL")
    print("-" * 60)

    if original_df is not None and customer_id_col in original_df.columns:
        # Create engagement segments based on total quotes
        if 'total_quotes' in original_df.columns:
            # Define engagement segments
            engagement_errors = pd.DataFrame({
                'customer_id': original_df[customer_id_col],
                'engagement_level': pd.cut(original_df['total_quotes'],
                                           bins=[0, 1, 2, 3, 100],
                                           labels=['Single', 'Low', 'Medium', 'High']),
                'actual': y_test,
                'predicted': y_pred,
                'is_error': (y_test != y_pred).astype(int)
            })

            # Error rates by engagement level
            engagement_error_rates = engagement_errors.groupby('engagement_level').agg(
                total_customers=('customer_id', 'count'),
                error_count=('is_error', 'sum'),
                fp_count=('is_error', lambda x: ((engagement_errors.loc[x.index, 'actual'] == 0) &
                                                 (engagement_errors.loc[x.index, 'predicted'] == 1)).sum()),
                fn_count=('is_error', lambda x: ((engagement_errors.loc[x.index, 'actual'] == 1) &
                                                 (engagement_errors.loc[x.index, 'predicted'] == 0)).sum())
            )

            engagement_error_rates['error_rate'] = engagement_error_rates['error_count'] / engagement_error_rates[
                'total_customers'] * 100
            engagement_error_rates['fp_rate'] = engagement_error_rates['fp_count'] / engagement_error_rates[
                'total_customers'] * 100
            engagement_error_rates['fn_rate'] = engagement_error_rates['fn_count'] / engagement_error_rates[
                'total_customers'] * 100

            print("Error Rates by Engagement Level:")
            print(engagement_error_rates[['total_customers', 'error_rate', 'fp_rate', 'fn_rate']].round(1).to_string())

            results['engagement_error_analysis'] = engagement_error_rates

            # Identify worst-performing engagement level
            worst_level = engagement_error_rates['error_rate'].idxmax()
            worst_error_rate = engagement_error_rates['error_rate'].max()

            print(f"\n⚠️  WORST PERFORMING ENGAGEMENT LEVEL:")
            print(f"  • {worst_level}: {worst_error_rate:.1f}% error rate")
            print(f"  • Needs specific feature engineering")

    # ========== 5. PATTERN ANALYSIS OF FALSE NEGATIVES ==========
    print(f"\n💡 5. FALSE NEGATIVE ANALYSIS (MISSED OPPORTUNITIES)")
    print("-" * 60)

    if 'error_type' in locals() and 'FN' in error_df['error_type'].values:
        fn_df = error_df[error_df['error_type'] == 'FN']

        if len(fn_df) > 0:
            print(f"Analyzing {len(fn_df):,} missed opportunities:")

            # Common patterns in false negatives
            patterns = []

            # Pattern 1: High consideration but predicted low
            if 'consideration_depth_score' in fn_df.columns:
                high_consideration_fn = (fn_df['consideration_depth_score'] > 0.6).sum()
                patterns.append(f"• {high_consideration_fn:,} had high consideration depth (>0.6)")

            # Pattern 2: Wrong process used
            if 'process_customer_fit' in fn_df.columns:
                poor_process_fit_fn = (fn_df['process_customer_fit'] < 0.4).sum()
                patterns.append(f"• {poor_process_fit_fn:,} had poor process fit (<0.4)")

            # Pattern 3: Multi-quote customers
            if 'total_quotes' in fn_df.columns:
                multi_quote_fn = (fn_df['total_quotes'] > 1).sum()
                patterns.append(f"• {multi_quote_fn:,} were multi-quote customers")

            # Pattern 4: Brand loyalty analysis
            if 'brand_loyalty_index' in fn_df.columns:
                loyal_fn = (fn_df['brand_loyalty_index'] > 0.7).sum()
                patterns.append(f"• {loyal_fn:,} had high brand loyalty (>0.7)")

            for pattern in patterns:
                print(f"  {pattern}")

            results['fn_patterns'] = patterns

    # ========== 6. RECOMMENDATIONS ==========
    print(f"\n🎯 6. ACTIONABLE RECOMMENDATIONS")
    print("=" * 60)

    recommendations = []

    # Based on error analysis
    if fp_rate > fn_rate:
        recommendations.append("Model is TOO AGGRESSIVE (too many False Positives)")
        recommendations.append("→ Increase prediction threshold")
        recommendations.append("→ Add features to identify 'tire-kickers'")
    else:
        recommendations.append("Model is TOO CONSERVATIVE (too many False Negatives)")
        recommendations.append("→ Decrease prediction threshold")
        recommendations.append("→ Improve features for high-potential customers")

    if 'low_confidence_count' in results and results['low_confidence_count'] > 0.1 * len(y_test):
        recommendations.append(
            f"Too many uncertain predictions ({results['low_confidence_count'] / len(y_test) * 100:.1f}%)")
        recommendations.append("→ Need better features for edge cases")
        recommendations.append("→ Consider human review for 40-60% confidence range")

    if 'worst_level' in locals():
        recommendations.append(f"Engagement level '{worst_level}' has highest error rate")
        recommendations.append(f"→ Create engagement-specific features")
        recommendations.append(f"→ Consider different thresholds for this group")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")

    return results


def analyze_feature_contribution_to_errors(model, X_test, y_test, y_pred):
    """
    Analyze which features contribute most to errors
    """
    print("\n🔬 FEATURE CONTRIBUTION TO ERRORS")
    print("-" * 60)

    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    # Analyze SHAP values if available
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Analyze SHAP for errors
        error_mask = (y_test != y_pred)
        if error_mask.any():
            print("\nSHAP Analysis for Errors:")
            error_shap = shap_values[error_mask]
            error_features = X_test.columns[np.abs(error_shap).mean(axis=0).argsort()[-5:][::-1]]

            print("Features with largest SHAP impact on errors:")
            for feat in error_features:
                print(f"  • {feat}")

    except ImportError:
        print("SHAP not available for detailed error analysis")
    except:
        print("Could not compute SHAP values")


def create_error_visualization(results, y_test, y_pred, y_pred_proba=None):
    """
    Create visualizations based on what data is available
    """

    # Determine what plots we can create
    plots_to_create = []

    # Always create confusion matrix
    plots_to_create.append('confusion_matrix')

    # Create confidence plot if we have probabilities
    if y_pred_proba is not None:
        plots_to_create.append('confidence_distribution')

    # Create engagement plot if we have the data
    if 'engagement_error_analysis' in results:
        plots_to_create.append('engagement_errors')

    # Create layout based on available plots
    n_plots = len(plots_to_create)

    if n_plots == 1:
        fig, axes = plt.subplots(figsize=(10, 8))
        axes = [axes]  # Make it iterable
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    elif n_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    # Create each plot
    for i, plot_type in enumerate(plots_to_create):
        if i >= len(axes):
            break

        if plot_type == 'confusion_matrix':
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title('Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        elif plot_type == 'confidence_distribution':
            axes[i].hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
            axes[i].set_title('Prediction Confidence')
            axes[i].set_xlabel('Probability')
            axes[i].set_ylabel('Count')
            axes[i].legend()

        elif plot_type == 'engagement_errors':
            engagement_data = results['engagement_error_analysis']
            engagement_data['error_rate'].plot(kind='bar', ax=axes[i], color='coral')
            axes[i].set_title('Error Rate by Engagement')
            axes[i].set_xlabel('Engagement Level')
            axes[i].set_ylabel('Error Rate (%)')
            axes[i].tick_params(axis='x', rotation=45)

    # Hide unused axes if any
    for i in range(len(plots_to_create), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
