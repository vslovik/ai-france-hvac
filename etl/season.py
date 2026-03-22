import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

from etl.price import remove_price_outliers


def visualize_conversion_by_season(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Seasonal Effects on Customer Conversion")
    print("=" * 80)

    # Ensure date column exists and is datetime
    if 'first_quote_date' not in customers.columns:
        print("Error: 'first_quote_date' column not found")
        return

    customers_clean = customers.copy()
    customers_clean['first_quote_date'] = pd.to_datetime(customers_clean['first_quote_date'])
    customers_clean['year'] = customers_clean['first_quote_date'].dt.year
    customers_clean['month'] = customers_clean['first_quote_date'].dt.month
    customers_clean['quarter'] = customers_clean['first_quote_date'].dt.quarter

    # Create season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    customers_clean['season'] = customers_clean['month'].apply(get_season)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seasonal Effects on Customer Conversion', fontsize=16, fontweight='bold')

    # Plot 1: Conversion by month
    ax1 = axes[0, 0]
    monthly_conv = customers_clean.groupby('month')['converted'].agg(['mean', 'count'])
    monthly_conv = monthly_conv.reindex(range(1, 13)).fillna(0)

    bars = ax1.bar(monthly_conv.index, monthly_conv['mean'] * 100,
                   color='steelblue', alpha=0.7)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Customer Conversion Rate (%)')
    ax1.set_title('Conversion Rate by Month')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample sizes (handle NaN)
    for i, (month, row) in enumerate(monthly_conv.iterrows()):
        if row['count'] > 0:
            ax1.text(month, row['mean'] * 100 + 1, f'n={int(row["count"]):,}',
                     ha='center', fontsize=8, rotation=45)

    # Plot 2: Conversion by season
    ax2 = axes[0, 1]
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_conv = customers_clean.groupby('season')['converted'].agg(['mean', 'count'])
    seasonal_conv = seasonal_conv.reindex(season_order).fillna(0)

    bars = ax2.bar(season_order, seasonal_conv['mean'] * 100,
                   color=['lightblue', 'lightgreen', 'orange', 'brown'], alpha=0.7)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Customer Conversion Rate (%)')
    ax2.set_title('Conversion Rate by Season')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, (season, row) in zip(bars, seasonal_conv.iterrows()):
        if row['count'] > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{row["mean"] * 100:.1f}%\nn={int(row["count"]):,}',
                     ha='center', fontweight='bold')

    # Plot 3: Product mix by season
    ax3 = axes[1, 0]
    products = ['Heat Pump', 'Boiler', 'Stove', 'AC']
    season_data = []

    for season in season_order:
        subset = customers_clean[customers_clean['season'] == season]
        product_shares = []
        for product in products:
            share = (subset['main_equipment_category'] == product).mean() * 100
            product_shares.append(share)
        season_data.append(product_shares)

    season_data = np.array(season_data).T

    bottom = np.zeros(len(season_order))
    for i, (product, shares) in enumerate(zip(products, season_data)):
        ax3.bar(season_order, shares, bottom=bottom, label=product, alpha=0.7)
        bottom += shares

    ax3.set_xlabel('Season')
    ax3.set_ylabel('Product Mix (%)')
    ax3.set_title('Product Mix by Season')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Heat pump conversion by season
    ax4 = axes[1, 1]
    hp_customers = customers_clean[customers_clean['ever_bought_heat_pump']]
    if len(hp_customers) > 0:
        hp_seasonal = hp_customers.groupby('season')['converted'].agg(['mean', 'count'])
        hp_seasonal = hp_seasonal.reindex(season_order).fillna(0)

        bars = ax4.bar(season_order, hp_seasonal['mean'] * 100,
                       color='green', alpha=0.7)
        ax4.set_xlabel('Season')
        ax4.set_ylabel('Heat Pump Conversion Rate (%)')
        ax4.set_title('Heat Pump Conversion by Season')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, (season, row) in zip(bars, hp_seasonal.iterrows()):
            if row['count'] > 0:
                ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                         f'{row["mean"] * 100:.1f}%\nn={int(row["count"]):,}',
                         ha='center', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No heat pump customers found',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Heat Pump Conversion (No Data)')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n" + "=" * 80)
    print("SEASONAL STATISTICS")
    print("=" * 80)

    print("\nConversion by season:")
    for season in season_order:
        subset = customers_clean[customers_clean['season'] == season]
        if len(subset) > 0:
            print(f"  {season}: {subset['converted'].mean():.1%} (n={len(subset):,})")

    # Statistical test for seasonality
    if len(customers_clean['season'].dropna()) > 0:
        from scipy.stats import chi2_contingency
        season_contingency = pd.crosstab(customers_clean['season'], customers_clean['converted'])
        chi2, p_value, dof, expected = chi2_contingency(season_contingency)
        print(f"\nSeasonality chi-square p-value: {p_value:.4f}")
        print(f"Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")


def visualize_regional_seasonality(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Regional Seasonal Pattern")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']

    # Focus on top regions
    top_regions = customers_clean['main_region'].value_counts().head(3).index.tolist()
    print(f"\nTop regions: {top_regions}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Seasonal Patterns by Region', fontsize=14, fontweight='bold')

    for idx, region in enumerate(top_regions):
        ax = axes[idx]
        subset = customers_clean[customers_clean['main_region'] == region]

        seasonal = subset.groupby('season')['converted'].mean() * 100
        seasonal = seasonal.reindex(season_order)

        ax.plot(season_order, seasonal.values, 'o-', linewidth=2, markersize=8)
        ax.set_ylim(30, 50)
        ax.set_xlabel('Season')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title(f'{region} (n={len(subset):,})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

