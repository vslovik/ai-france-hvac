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

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    susp_conv = customers_clean.groupby('during_suspension')['converted'].agg(['mean', 'count'])
    susp_conv.index = ['Normal Periods', 'During Suspension']

    # Best and worst months
    monthly_stats = customers_clean.groupby('month')['converted'].agg(['mean', 'count'])
    best_month = monthly_stats['mean'].idxmax()
    worst_month = monthly_stats['mean'].idxmin()
    print(
        f"\nBest month: {best_month} ({monthly_stats.loc[best_month, 'mean']:.1%}, n={monthly_stats.loc[best_month, 'count']:.0f})")
    print(
        f"Worst month: {worst_month} ({monthly_stats.loc[worst_month, 'mean']:.1%}, n={monthly_stats.loc[worst_month, 'count']:.0f})")

    # Best and worst seasons
    seasonal_stats = customers_clean.groupby('season')['converted'].agg(['mean', 'count'])
    best_season = seasonal_stats['mean'].idxmax()
    worst_season = seasonal_stats['mean'].idxmin()
    print(f"\nBest season: {best_season} ({seasonal_stats.loc[best_season, 'mean']:.1%})")
    print(f"Worst season: {worst_season} ({seasonal_stats.loc[worst_season, 'mean']:.1%})")

    # Suspension impact
    print(
        f"\nSuspension periods impact: {susp_conv.loc['During Suspension', 'mean']:.1%} vs {susp_conv.loc['Normal Periods', 'mean']:.1%}")
    print(
        f"Difference: {(susp_conv.loc['During Suspension', 'mean'] - susp_conv.loc['Normal Periods', 'mean']) * 100:.1f} points")

    # Heat pump seasonality
    hp_seasonal = customers_clean[customers_clean['ever_bought_heat_pump']].groupby('season')['converted'].mean()
    print(f"\nHeat pump conversion by season:")
    for season in season_order:
        if season in hp_seasonal.index:
            print(f"  {season}: {hp_seasonal[season]:.1%}")


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

