import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Any


def get_top_decile_lift(y_true, y_pred_proba, n_deciles=10):
    """
    Calculate lift in top decile.
    """
    # Create dataframe
    df = pd.DataFrame({'actual': y_true, 'prob': y_pred_proba})

    # Sort by probability descending
    df = df.sort_values('prob', ascending=False)

    # Assign deciles using rank
    df['decile'] = pd.qcut(df['prob'].rank(method='first'),
                           q=n_deciles, labels=False) + 1

    # Calculate lift
    baseline = df['actual'].mean()
    decile1_rate = df[df['decile'] == 1]['actual'].mean()

    return decile1_rate / baseline if baseline > 0 else 0


def create_lift_table(
        df: pd.DataFrame,
        model: Any,
        features: List[str],
        target: str = 'target',
        n_bins: int = 10
) -> pd.DataFrame:
    """
    Create detailed lift table for analysis.
    """
    X = df[features]
    y = df[target]

    y_pred_proba = model.predict_proba(X)[:, 1]

    # Create results dataframe
    results_df = pd.DataFrame({
        'actual': y,
        'predicted_prob': y_pred_proba
    })

    # ✅ FIX 2: Sort descending and assign deciles manually
    results_df = results_df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)

    # Manually assign deciles
    decile_size = len(results_df) // n_bins
    results_df['decile'] = 0

    for i in range(n_bins):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_bins - 1 else len(results_df)
        results_df.loc[start_idx:end_idx - 1, 'decile'] = i + 1

    # Verify the fix
    print("Verification of decile assignment:")
    print(f"Decile 1 (top) mean probability: {results_df[results_df['decile'] == 1]['predicted_prob'].mean():.3f}")
    print(
        f"Decile {n_bins} (bottom) mean probability: {results_df[results_df['decile'] == n_bins]['predicted_prob'].mean():.3f}")

    # Calculate metrics
    lift_data = []
    cumulative_positives = 0
    cumulative_total = 0

    for decile in range(1, n_bins + 1):
        decile_data = results_df[results_df['decile'] == decile]
        n_in_decile = len(decile_data)
        positives = decile_data['actual'].sum()

        cumulative_positives += positives
        cumulative_total += n_in_decile

        # Rates
        response_rate = (positives / n_in_decile) * 100
        capture_rate = (positives / results_df['actual'].sum()) * 100
        cumulative_capture = (cumulative_positives / results_df['actual'].sum()) * 100

        # Lift calculations
        overall_response_rate = (results_df['actual'].sum() / len(results_df)) * 100
        lift = response_rate / overall_response_rate if overall_response_rate > 0 else 0
        cumulative_lift = ((cumulative_positives / cumulative_total) * 100) / overall_response_rate

        lift_data.append({
            'Decile': decile,
            'N': n_in_decile,
            '# Positives': positives,
            'Response Rate %': f"{response_rate:.1f}%",
            'Cum Response Rate %': f"{(cumulative_positives / cumulative_total) * 100:.1f}%",
            'Capture Rate %': f"{capture_rate:.1f}%",
            'Cum Capture %': f"{cumulative_capture:.1f}%",
            'Lift': f"{lift:.2f}",
            'Cumulative Lift': f"{cumulative_lift:.2f}",
            'Gain over Random': f"{(lift - 1) * 100:.1f}%"
        })

    return pd.DataFrame(lift_data)


def plot_lift_analysis(y_true, y_pred_proba, n_deciles=10):
    """
    Plot lift chart and cumulative gains.
    """
    df = pd.DataFrame({'actual': y_true, 'prob': y_pred_proba})

    # ✅ ONLY ONE METHOD - Choose one:

    # Option A: Simple and reliable
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(df.index, q=n_deciles, labels=False) + 1

    # OR Option B: Using rank (also works)
    # df = df.sort_values('prob', ascending=False)
    # df['decile'] = pd.qcut(df['prob'].rank(method='first'),
    #                       q=n_deciles, labels=False) + 1

    # ✅ VERIFY (keep this)
    print("Verification of decile assignment:")
    print(f"Decile 1 mean prob: {df[df['decile'] == 1]['prob'].mean():.3f}")
    print(f"Decile 10 mean prob: {df[df['decile'] == 10]['prob'].mean():.3f}")

    if df[df['decile'] == 1]['prob'].mean() < df[df['decile'] == 10]['prob'].mean():
        print("⚠️ WARNING: Deciles inverted! Flipping...")
        df['decile'] = (n_deciles + 1) - df['decile']

    # Calculate metrics per decile
    decile_stats = df.groupby('decile').agg(
        count=('actual', 'size'),
        positives=('actual', 'sum'),
        mean_prob=('prob', 'mean')
    ).reset_index()

    baseline = df['actual'].mean()
    decile_stats['response_rate'] = decile_stats['positives'] / decile_stats['count']
    decile_stats['lift'] = decile_stats['response_rate'] / baseline

    # Cumulative metrics
    decile_stats['cum_population'] = decile_stats['count'].cumsum() / len(df) * 100
    decile_stats['cum_positives'] = decile_stats['positives'].cumsum()
    decile_stats['cum_positives_pct'] = decile_stats['cum_positives'] / df['actual'].sum() * 100

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Lift Chart
    colors = ['green' if lift > 1 else 'red' for lift in decile_stats['lift']]
    axes[0].bar(decile_stats['decile'], decile_stats['lift'], color=colors, alpha=0.7)
    axes[0].axhline(y=1, color='black', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Decile (1 = Highest Score)')
    axes[0].set_ylabel('Lift')
    axes[0].set_title(f'Lift Chart\nTop Decile: {decile_stats.iloc[0]["lift"]:.2f}x')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Cumulative Gains
    axes[1].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
    axes[1].plot(decile_stats['cum_population'], decile_stats['cum_positives_pct'],
                 'b-o', linewidth=2, label='Model')
    axes[1].set_xlabel('% of Population')
    axes[1].set_ylabel('% of Positives Captured')
    axes[1].set_title(f'Cumulative Gains\nTop 30% → {decile_stats.iloc[2]["cum_positives_pct"]:.1f}% captured')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Response Rate
    axes[2].bar(decile_stats['decile'], decile_stats['response_rate'] * 100,
                alpha=0.6, color='orange', label='Response Rate')
    axes[2].axhline(y=baseline * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Baseline: {baseline * 100:.1f}%')
    axes[2].set_xlabel('Decile (1 = Highest Score)')
    axes[2].set_ylabel('Response Rate (%)')
    axes[2].set_title('Response Rate by Decile')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Business Impact Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return decile_stats


def plot_lift_chart(
        df: pd.DataFrame,
        model: Any,
        features: List[str],
        target: str = 'target',
        n_bins: int = 10,
        figsize: Tuple[int, int] = (15, 5),
        title: str = 'Model Lift Analysis'
) -> Dict[str, Any]:
    """
    Plot lift chart and related visualizations.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    model : Any
        Trained model with predict_proba method
    features : List[str]
        Feature column names
    target : str
        Target column name
    n_bins : int
        Number of bins/deciles (default: 10)
    figsize : Tuple[int, int]
        Figure size
    title : str
        Plot title

    Returns:
    --------
    Dict: Lift metrics and data
    """

    """
    Create detailed lift table for analysis.
    """
    X = df[features]
    y = df[target]

    y_pred_proba = model.predict_proba(X)[:, 1]

    # Create results dataframe
    results_df = pd.DataFrame({
        'actual': y,
        'predicted_prob': y_pred_proba
    })

    # ✅ FIX 2: Sort descending and assign deciles manually
    results_df = results_df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)

    # Manually assign deciles
    decile_size = len(results_df) // n_bins
    results_df['decile'] = 0

    for i in range(n_bins):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_bins - 1 else len(results_df)
        results_df.loc[start_idx:end_idx - 1, 'decile'] = i + 1

    # Verify decile assignment
    print("Decile assignment verification:")
    for i in range(1, min(4, n_bins + 1)):
        decile_data = results_df[results_df['decile'] == i]
        print(f"  Decile {i}: Mean prob = {decile_data['predicted_prob'].mean():.4f}, "
              f"Response rate = {decile_data['actual'].mean():.2%}")

    # Calculate metrics per decile
    lift_data = []

    for decile in range(1, n_bins + 1):
        decile_df = results_df[results_df['decile'] == decile]
        n_in_decile = len(decile_df)
        positives_in_decile = decile_df['actual'].sum()

        # Cumulative metrics
        cumulative_df = results_df[results_df['decile'] <= decile]
        cumulative_positives = cumulative_df['actual'].sum()

        # Calculate rates
        capture_rate = (positives_in_decile / len(results_df[results_df['actual'] == 1])) * 100
        cumulative_capture_rate = (cumulative_positives / len(results_df[results_df['actual'] == 1])) * 100
        population_percentage = (n_in_decile / len(results_df)) * 100
        cumulative_population = (len(cumulative_df) / len(results_df)) * 100

        # Calculate lift
        expected_rate = (len(results_df[results_df['actual'] == 1]) / len(results_df)) * 100
        lift = (positives_in_decile / n_in_decile * 100) / expected_rate if expected_rate > 0 else 0

        lift_data.append({
            'decile': decile,
            'population_%': population_percentage,
            'cumulative_population_%': cumulative_population,
            'positives_count': positives_in_decile,
            'cumulative_positives': cumulative_positives,
            'capture_rate_%': capture_rate,
            'cumulative_capture_%': cumulative_capture_rate,
            'response_rate_%': (positives_in_decile / n_in_decile) * 100,
            'lift': lift
        })

    lift_df = pd.DataFrame(lift_data)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. CUMULATIVE GAINS CHART (Lift Chart)
    ax1 = axes[0]

    # Perfect model line (diagonal from 0,0 to 100,100)
    ax1.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random Model')

    # Actual model curve
    ax1.plot(lift_df['cumulative_population_%'],
             lift_df['cumulative_capture_%'],
             'b-o', linewidth=2, markersize=8,
             label='Our Model')

    # Perfect model (if we could perfectly rank)
    n_positives = len(results_df[results_df['actual'] == 1])
    total_samples = len(results_df)
    perfect_x = np.linspace(0, 100, 100)
    perfect_y = [min(100, (x / 100 * total_samples) / n_positives * 100)
                 for x in perfect_x]
    ax1.plot(perfect_x, perfect_y, 'g--', alpha=0.7, label='Perfect Model')

    ax1.set_xlabel('% of Population', fontsize=12)
    ax1.set_ylabel('% of Positives Captured', fontsize=12)
    ax1.set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 105])

    # Add lift values as text
    for i, row in lift_df.iterrows():
        if i % 2 == 0:  # Show every other decile to avoid clutter
            ax1.text(row['cumulative_population_%'] - 5,
                     row['cumulative_capture_%'] + 2,
                     f"Lift: {row['lift']:.2f}",
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="yellow", alpha=0.7))

    # 2. LIFT CHART PER DECILE
    ax2 = axes[1]

    # Bar chart of lift per decile
    colors = ['green' if lift > 1 else 'red' for lift in lift_df['lift']]
    bars = ax2.bar(lift_df['decile'], lift_df['lift'], color=colors, alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5,
                label='Baseline (Lift = 1)')
    ax2.set_xlabel('Decile (1 = Highest Score)', fontsize=12)
    ax2.set_ylabel('Lift', fontsize=12)
    ax2.set_title('Lift by Decile', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(1, n_bins + 1))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. RESPONSE RATE CHART
    ax3 = axes[2]

    # Calculate baseline response rate
    baseline_rate = (len(results_df[results_df['actual'] == 1]) / len(results_df)) * 100

    # Bar chart of response rate
    ax3.bar(lift_df['decile'], lift_df['response_rate_%'],
            alpha=0.6, color='orange', label='Response Rate')
    ax3.axhline(y=baseline_rate, color='red', linestyle='--',
                linewidth=2, label=f'Baseline: {baseline_rate:.1f}%')

    ax3.set_xlabel('Decile (1 = Highest Score)', fontsize=12)
    ax3.set_ylabel('Response Rate (%)', fontsize=12)
    ax3.set_title('Response Rate by Decile', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(1, n_bins + 1))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=" * 80)
    print(f"{'LIFT ANALYSIS SUMMARY':^70}")
    print("=" * 80)
    print(f"Total samples: {len(results_df):,}")
    print(f"Positive cases: {n_positives:,} ({n_positives / len(results_df) * 100:.1f}%)")
    print(f"Baseline response rate: {baseline_rate:.2f}%")
    print("\nTop Deciles Performance:")
    print("-" * 40)

    for i in range(min(3, len(lift_df))):
        decile_data = lift_df.iloc[i]
        print(f"Decile {int(decile_data['decile'])}:")
        print(f"  • Lift: {decile_data['lift']:.2f}x baseline")
        print(f"  • Response rate: {decile_data['response_rate_%']:.1f}%")
        print(f"  • Captures {decile_data['capture_rate_%']:.1f}% of all positives")

    print(f"\nTop 3 deciles capture: {lift_df.iloc[2]['cumulative_capture_%']:.1f}% of positives")
    print(f"from only {lift_df.iloc[2]['cumulative_population_%']:.1f}% of population")

    # ✅ ADDED: Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE ASSESSMENT:")
    print("-" * 80)

    top_decile_lift = lift_df.iloc[0]['lift']
    if top_decile_lift > 2.0:
        print(f"✅ EXCELLENT: Top decile lift = {top_decile_lift:.2f}x (>2.0x)")
    elif top_decile_lift > 1.5:
        print(f"👍 GOOD: Top decile lift = {top_decile_lift:.2f}x (1.5-2.0x)")
    elif top_decile_lift > 1.0:
        print(f"⚠️  WEAK: Top decile lift = {top_decile_lift:.2f}x (1.0-1.5x)")
    else:
        print(f"❌ POOR: Top decile lift = {top_decile_lift:.2f}x (<1.0x)")

    top3_capture = lift_df.iloc[2]['cumulative_capture_%']
    if top3_capture > 60:
        print(f"✅ EXCELLENT: Top 3 deciles capture {top3_capture:.1f}% of positives")
    elif top3_capture > 40:
        print(f"👍 GOOD: Top 3 deciles capture {top3_capture:.1f}% of positives")
    else:
        print(f"⚠️  WEAK: Top 3 deciles capture only {top3_capture:.1f}% of positives")

    return {
        'lift_data': lift_df,
        'predictions': results_df,
        'baseline_rate': baseline_rate,
        'total_positives': n_positives,
        'total_samples': len(results_df)
    }