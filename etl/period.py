import pandas as pd
from matplotlib import pyplot as plt

from etl.price import remove_price_outliers


def visualize_conversion_by_year(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Year-over-year trends")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])

    # Group by year and quarter
    customers_clean['year_quarter'] = customers_clean['first_quote_date'].dt.to_period('Q')
    yearly_trend = customers_clean.groupby('year_quarter')['converted'].agg(['mean', 'count'])

    print("\nQuarterly conversion rates:")
    print(yearly_trend)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Quarterly trend
    ax1 = axes[0]
    quarters = [str(q) for q in yearly_trend.index]
    values = yearly_trend['mean'] * 100

    ax1.plot(quarters, values, 'bo-', linewidth=2, markersize=8)
    ax1.fill_between(quarters, values - 2, values + 2, alpha=0.2, color='blue')
    ax1.set_xlabel('Year-Quarter')
    ax1.set_ylabel('Customer Conversion Rate (%)')
    ax1.set_title('Quarterly Conversion Trend')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Volume trend
    ax2 = axes[1]
    volumes = yearly_trend['count'].values
    ax2.bar(quarters, volumes, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Year-Quarter')
    ax2.set_ylabel('Number of New Customers')
    ax2.set_title('Customer Volume by Quarter')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def report_best_and_worst_months(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Best and Worst Months Report")
    print("=" * 80)

    # Remove outliers if needed
    customers_clean = remove_price_outliers(customers, price_var=price_var)

    # Check during_suspension unique values
    unique_susp = customers_clean['during_suspension'].unique()

    # Only create suspension comparison if both values exist
    if len(unique_susp) == 2:
        susp_conv = customers_clean.groupby('during_suspension')['converted'].agg(['mean', 'count'])
        susp_conv.index = ['Normal Periods', 'During Suspension']
        print("\nConversion during subsidy suspensions:")
        print(susp_conv)

        # Statistical test
        from scipy.stats import chi2_contingency
        susp_contingency = pd.crosstab(customers_clean['during_suspension'], customers_clean['converted'])
        chi2, p_value, dof, expected = chi2_contingency(susp_contingency)
        print(f"\nSuspension impact p-value: {p_value:.4f}")
        print(f"Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")
    else:
        print(f"\n⚠️  Only one suspension period found: {unique_susp}")
        print("Skipping suspension comparison")

    # Best and worst months
    monthly_stats = customers_clean.groupby('month')['converted'].agg(['mean', 'count'])
    monthly_stats = monthly_stats.reindex(range(1, 13)).fillna(0)

    best_month = monthly_stats['mean'].idxmax() if monthly_stats['mean'].max() > 0 else None
    worst_month = monthly_stats['mean'].idxmin() if monthly_stats['mean'].min() > 0 else None

    if best_month:
        print(
            f"\nBest month: {best_month} ({monthly_stats.loc[best_month, 'mean']:.1%}, n={monthly_stats.loc[best_month, 'count']:.0f})")
    if worst_month:
        print(
            f"Worst month: {worst_month} ({monthly_stats.loc[worst_month, 'mean']:.1%}, n={monthly_stats.loc[worst_month, 'count']:.0f})")

    # Best and worst seasons
    seasonal_stats = customers_clean.groupby('season')['converted'].agg(['mean', 'count'])

    best_season = seasonal_stats['mean'].idxmax() if seasonal_stats['mean'].max() > 0 else None
    worst_season = seasonal_stats['mean'].idxmin() if seasonal_stats['mean'].min() > 0 else None

    if best_season:
        print(f"\nBest season: {best_season} ({seasonal_stats.loc[best_season, 'mean']:.1%})")
    if worst_season:
        print(f"Worst season: {worst_season} ({seasonal_stats.loc[worst_season, 'mean']:.1%})")

    # Heat pump seasonality
    hp_customers = customers_clean[customers_clean['ever_bought_heat_pump']]
    if len(hp_customers) > 0:
        hp_seasonal = hp_customers.groupby('season')['converted'].mean()
        print(f"\nHeat pump conversion by season:")
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            if season in hp_seasonal.index:
                print(f"  {season}: {hp_seasonal[season]:.1%}")

    return monthly_stats, seasonal_stats


def show_conversion_by_price_over_time(customers, price_var='max_out_of_pocket'):
    customers_clean = remove_price_outliers(customers, price_var=price_var)
    # Convert first_quote_date to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(customers_clean['first_quote_date']):
        customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])
    customers_clean['period'] = customers_clean['first_quote_date'].dt.year.astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price-Conversion Relationship: How It Has Changed Over Time', fontsize=16, fontweight='bold')

    periods = ['2023', '2024', '2025', '2026']
    colors = ['green', 'blue', 'orange', 'red']

    for idx, (period, color) in enumerate(zip(periods, colors)):
        ax = axes[idx // 2, idx % 2]

        subset = customers_clean[customers_clean['period'] == period]
        if len(subset) > 200:
            # Create price bins
            try:
                subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=15, duplicates='drop')
                bin_conv = subset.groupby('price_bin')['converted'].mean()
                bin_price = subset.groupby('price_bin')['max_out_of_pocket'].mean()

                ax.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2, markersize=4)
                ax.axhline(y=subset['converted'].mean(), color=color, linestyle='--',
                           alpha=0.5, label=f'Avg: {subset["converted"].mean():.1%}')

                # Highlight heat pump range
                ax.axvspan(12000, 16000, alpha=0.1, color='green', label='Heat pump range')

                ax.set_xlabel('Maximum Quote Price (€)')
                ax.set_ylabel('Conversion Rate')
                ax.set_title(f'{period} (n={len(subset):,}, overall conv={subset["converted"].mean():.1%})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_ylim(0, 0.7)
            except:
                ax.text(0.5, 0.5, f'Insufficient data for {period}', ha='center', va='center')
                ax.set_title(f'{period} (insufficient data)')

    plt.tight_layout()
    plt.show()


def overlay_all_years_on_one_plot(customers, price_var='max_out_of_pocket'):
    customers_clean = remove_price_outliers(customers, price_var=price_var)
    # Convert first_quote_date to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(customers_clean['first_quote_date']):
        customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])
    customers_clean['period'] = customers_clean['first_quote_date'].dt.year.astype(str)
    # Overlay all years on one plot
    fig, ax = plt.subplots(figsize=(14, 8))

    for period, color in zip(['2023', '2024', '2025', '2026'], ['green', 'blue', 'orange', 'red']):
        subset = customers_clean[customers_clean['period'] == period]
        if len(subset) > 200:
            # Create price bins
            subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=10, duplicates='drop')
            bin_conv = subset.groupby('price_bin')['converted'].mean()
            bin_price = subset.groupby('price_bin')['max_out_of_pocket'].mean()

            ax.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2,
                    label=f'{period} (n={len(subset):,}, conv={subset["converted"].mean():.1%})')

    ax.axvspan(12000, 16000, alpha=0.1, color='green', label='Heat pump range')
    ax.set_xlabel('Maximum Quote Price (€)')
    ax.set_ylabel('Customer Conversion Rate')
    ax.set_title('The Conversion Collapse: Price-Response Curves by Year')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.7)

    plt.tight_layout()
    plt.show()


def visualize_conversion_by_prices_sweet_spot_shift(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Sweet Spot Analysis Over Time")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    # Convert first_quote_date to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(customers_clean['first_quote_date']):
        customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])
    customers_clean['period'] = customers_clean['first_quote_date'].dt.year.astype(str)

    # Function to identify sweet spots
    def find_sweet_spots(df, year, n_bins=20):
        subset = df[df['period'] == year].copy()
        if len(subset) < 500:
            return None

        subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=n_bins, duplicates='drop')
        bin_stats = subset.groupby('price_bin').agg({
            'converted': ['mean', 'count'],
            'max_out_of_pocket': 'mean'
        })
        bin_stats.columns = ['conv_rate', 'count', 'price']
        bin_stats = bin_stats.reset_index()

        overall_avg = subset['converted'].mean()
        sweet_spots = bin_stats[bin_stats['conv_rate'] > overall_avg + 0.03].copy()  # 3 points above avg

        return sweet_spots

    print("\nSweet spots by year (price points with conversion > avg + 3%):")
    for year in ['2023', '2024', '2025']:
        sweet = find_sweet_spots(customers_clean, year)
        if sweet is not None and len(sweet) > 0:
            print(
                f"\n{year} (overall avg: {customers_clean[customers_clean['period'] == year]['converted'].mean():.1%}):")
            for _, row in sweet.iterrows():
                print(f"  €{row['price']:,.0f}: {row['conv_rate']:.1%} (n={row['count']:.0f})")

    # Visualization of shifting sweet spots
    fig, ax = plt.subplots(figsize=(14, 8))

    for year, color in zip(['2023', '2024', '2025'], ['green', 'blue', 'orange']):
        subset = customers_clean[customers_clean['period'] == year].copy()
        if len(subset) > 500:
            subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=20, duplicates='drop')
            bin_stats = subset.groupby('price_bin').agg({
                'converted': 'mean',
                'max_out_of_pocket': 'mean'
            }).reset_index(drop=True)

            # Smooth for better visualization
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(bin_stats['converted'].values, sigma=1)

            ax.plot(bin_stats['max_out_of_pocket'], smoothed, color=color, linewidth=2,
                    label=f'{year} (avg: {subset["converted"].mean():.1%})')

            # Mark the peak
            peak_idx = smoothed.argmax()
            peak_price = bin_stats['max_out_of_pocket'].iloc[peak_idx]
            peak_conv = smoothed[peak_idx]
            ax.plot(peak_price, peak_conv, 'o', color=color, markersize=8)
            ax.text(peak_price, peak_conv + 0.02, f'€{peak_price:,.0f}\n{peak_conv:.1%}',
                    ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Maximum Quote Price (€)')
    ax.set_ylabel('Conversion Rate (smoothed)')
    ax.set_title('Sweet Spots Are Shifting: Where Should You Focus?')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_decision_times(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Administrative Uncertainty Effect")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    # Convert first_quote_date to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(customers_clean['first_quote_date']):
        customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])
    customers_clean['period'] = customers_clean['first_quote_date'].dt.year.astype(str)

    # First, let's examine the decision time distribution
    print("\nDecision time distribution:")
    print(customers_clean['decision_days'].describe())

    # Check how many have 1-day decisions
    one_day_count = (customers_clean['decision_days'] == 1).sum()
    print(f"\nCustomers with 1-day decision: {one_day_count:,} ({one_day_count / len(customers_clean):.1%})")

    # Fix: Create custom bins instead of quantile bins
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Administrative Uncertainty: Does Decision Time Matter?', fontsize=16, fontweight='bold')

    # Plot 1: Conversion by decision time bins (custom bins)
    ax1 = axes[0, 0]
    # Create meaningful time bins
    bins = [0, 1, 7, 30, 90, 365]
    labels = ['Same day (1)', 'Quick (2-7 days)', 'Medium (8-30 days)', 'Slow (31-90 days)', 'Very Slow (>90 days)']

    customers_clean['time_bin'] = pd.cut(customers_clean['decision_days'], bins=bins, labels=labels)

    time_conv = customers_clean.groupby('time_bin')['converted'].mean() * 100
    time_counts = customers_clean.groupby('time_bin').size()

    bars = ax1.bar(range(len(time_conv)), time_conv.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(time_conv)))
    ax1.set_xticklabels(time_conv.index, rotation=45, ha='right')
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Conversion by Decision Time')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (bar, (idx, count)) in enumerate(zip(bars, time_counts.items())):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%\nn={count:,}', ha='center', fontsize=9)

    # Plot 2: Decision time by year (is it getting longer?)
    ax2 = axes[0, 1]
    yearly_time = customers_clean.groupby('period')['decision_days'].median()
    ax2.plot(yearly_time.index, yearly_time.values, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Median Decision Time (days)')
    ax2.set_title('Decision Time Getting Longer?')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Price volatility by conversion status
    ax3 = axes[1, 0]
    conv_cv = customers_clean[customers_clean['converted'] == 1]['price_cv'].dropna()
    nonconv_cv = customers_clean[customers_clean['converted'] == 0]['price_cv'].dropna()

    ax3.boxplot([nonconv_cv, conv_cv], labels=['Not Converted', 'Converted'])
    ax3.set_ylabel('Price Volatility (CV)')
    ax3.set_title('Price Volatility: Converted vs Not Converted')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Quote count by conversion status
    ax4 = axes[1, 1]
    conv_quotes = customers_clean[customers_clean['converted'] == 1]['quote_count']
    nonconv_quotes = customers_clean[customers_clean['converted'] == 0]['quote_count']

    ax4.boxplot([nonconv_quotes, conv_quotes], labels=['Not Converted', 'Converted'])
    ax4.set_ylabel('Number of Quotes')
    ax4.set_title('Quote Count: Converted vs Not Converted')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def show_decision_time_dashboard(customers, price_var='max_out_of_pocket'):

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    # Convert first_quote_date to datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(customers_clean['first_quote_date']):
        customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])
    customers_clean['period'] = customers_clean['first_quote_date'].dt.year.astype(str)

    # Create the definitive administrative burden visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Most Customers Decide in 1 Day', fontsize=16, fontweight='bold')

    # Plot 1: The 1-day dominance
    # Plot 1: The 1-day dominance (DYNAMIC with error handling)
    ax1 = axes[0, 0]

    # Calculate dynamically from data with safe fallbacks
    if 'decision_days' in customers_clean.columns:
        same_day_count = (customers_clean['decision_days'] == 1).sum()
        longer_count = (customers_clean['decision_days'] > 1).sum()
        total_customers = len(customers_clean)

        same_day_pct = (same_day_count / total_customers) * 100
        longer_pct = (longer_count / total_customers) * 100

        labels = [f'Same Day Decision\n({same_day_count:,} customers)',
                  f'Longer Process\n({longer_count:,} customers)']
        sizes = [same_day_pct, longer_pct]

        # Only create pie chart if we have valid data
        if total_customers > 0 and same_day_pct + longer_pct > 99:
            colors = ['green', 'orange']
            explode = (0.1, 0)

            wedges, texts, autotexts = ax1.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct=lambda pct: f'{pct:.0f}%\n({int(pct * total_customers / 100):,})',
                shadow=True,
                startangle=90
            )

            # Make the percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')

            ax1.set_title(f'{same_day_pct:.0f}% of Customers Decide in 1 Day', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No decision_days data available',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Decision Time Analysis (Data Missing)')

    # Plot 2: Conversion by decision time
    ax2 = axes[0, 1]
    time_bins = ['Same Day', '2-7 days', '8-30 days', '31-90 days', '>90 days']
    conv_by_time = [
        customers_clean[customers_clean['decision_days'] == 1]['converted'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 1) & (customers_clean['decision_days'] <= 7)][
            'converted'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 7) & (customers_clean['decision_days'] <= 30)][
            'converted'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 30) & (customers_clean['decision_days'] <= 90)][
            'converted'].mean() * 100,
        customers_clean[customers_clean['decision_days'] > 90]['converted'].mean() * 100
    ]
    bars = ax2.bar(time_bins, conv_by_time, color=['green', 'yellowgreen', 'orange', 'red', 'darkred'])
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_title('Longer Process = Higher Conversion?')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, conv_by_time):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{val:.1f}%', ha='center')

    # Plot 3: Heat pump share by decision time
    ax3 = axes[1, 0]
    hp_share = [
        customers_clean[customers_clean['decision_days'] == 1]['ever_bought_heat_pump'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 1) & (customers_clean['decision_days'] <= 7)][
            'ever_bought_heat_pump'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 7) & (customers_clean['decision_days'] <= 30)][
            'ever_bought_heat_pump'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 30) & (customers_clean['decision_days'] <= 90)][
            'ever_bought_heat_pump'].mean() * 100,
        customers_clean[customers_clean['decision_days'] > 90]['ever_bought_heat_pump'].mean() * 100
    ]
    ax3.plot(time_bins, hp_share, 'go-', linewidth=2, markersize=8)
    ax3.set_ylabel('Heat Pump Share (%)')
    ax3.set_title('Heat Pumps Dominate Long Sales Cycles')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Subsidy issues by decision time
    ax4 = axes[1, 1]
    subsidy_share = [
        customers_clean[customers_clean['decision_days'] == 1]['had_subsidy_issue'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 1) & (customers_clean['decision_days'] <= 7)][
            'had_subsidy_issue'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 7) & (customers_clean['decision_days'] <= 30)][
            'had_subsidy_issue'].mean() * 100,
        customers_clean[(customers_clean['decision_days'] > 30) & (customers_clean['decision_days'] <= 90)][
            'had_subsidy_issue'].mean() * 100,
        customers_clean[customers_clean['decision_days'] > 90]['had_subsidy_issue'].mean() * 100
    ]
    ax4.bar(time_bins, subsidy_share, color='purple', alpha=0.7)
    ax4.set_ylabel('Subsidy Issue Rate (%)')
    ax4.set_title('Long Cycles = Subsidy Complexity')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

