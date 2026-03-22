import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def remove_price_outliers(customers, price_var='max_out_of_pocket'):
    # Use max_out_of_pocket as the key price variable (highest quote they considered)

    # Remove extreme outliers for clean visualization
    price_lower = customers[price_var].quantile(0.01)
    price_upper = customers[price_var].quantile(0.99)
    customers_clean = customers[(customers[price_var] >= price_lower) &
                                (customers[price_var] <= price_upper)].copy()

    print(f"\nAnalyzing {len(customers_clean):,} customers")
    print(f"Price range: €{customers_clean[price_var].min():,.0f} to €{customers_clean[price_var].max():,.0f}")
    print(f"Median price: €{customers_clean[price_var].median():,.0f}")

    return customers_clean


def price_binned_stats(customers_clean, price_var='max_out_of_pocket'):
    # Create price bins with equal number of customers
    customers_clean['price_bin'] = pd.qcut(customers_clean[price_var], q=30, duplicates='drop')
    # Calculate conversion rate per bin
    binned_stats = customers_clean.groupby('price_bin').agg({
        'converted': ['mean', 'count', 'sem'],
        price_var: 'mean'
    }).round(4)

    binned_stats.columns = ['conversion_rate', 'customer_count', 'std_error', 'avg_price']
    binned_stats['ci_lower'] = binned_stats['conversion_rate'] - 1.96 * binned_stats['std_error']
    binned_stats['ci_upper'] = binned_stats['conversion_rate'] + 1.96 * binned_stats['std_error']
    binned_stats = binned_stats.reset_index()

    return binned_stats


def visualize_conversion_by_price_sweet_spots(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Threshold effect")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)
    binned_stats = price_binned_stats(customers_clean, price_var=price_var)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price Threshold Effects (Customer-Level Conversion)', fontsize=16, fontweight='bold')

    # Plot 1: Binned conversion with confidence intervals
    ax1 = axes[0, 0]
    ax1.errorbar(binned_stats['avg_price'], binned_stats['conversion_rate'],
                 yerr=1.96 * binned_stats['std_error'],
                 fmt='o', color='steelblue', ecolor='lightgray',
                 capsize=3, markersize=4, alpha=0.7)
    ax1.axhline(y=customers_clean['converted'].mean(), color='red',
                linestyle='--', alpha=0.5, label=f"Overall avg: {customers_clean['converted'].mean():.1%}")
    ax1.set_xlabel('Maximum Quote Price (€)')
    ax1.set_ylabel('Customer Conversion Rate')
    ax1.set_title('Conversion Rate by Price (with 95% CI)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Smoothed curve
    ax2 = axes[0, 1]
    smoothed = binned_stats['conversion_rate'].rolling(window=5, center=True).mean()

    ax2.plot(binned_stats['avg_price'], binned_stats['conversion_rate'],
             'o', alpha=0.3, markersize=3, label='Raw bins')
    ax2.plot(binned_stats['avg_price'], smoothed, 'b-', linewidth=3, label='Smoothed')
    ax2.axhline(y=customers_clean['converted'].mean(), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Maximum Quote Price (€)')
    ax2.set_ylabel('Customer Conversion Rate')
    ax2.set_title('Smoothed Price-Conversion Relationship')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Volume distribution
    ax3 = axes[1, 0]
    ax3.bar(binned_stats['avg_price'], binned_stats['customer_count'],
            width=np.diff(binned_stats['avg_price']).mean() * 0.8,
            color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax3.set_xlabel('Maximum Quote Price (€)')
    ax3.set_ylabel('Number of Customers')
    ax3.set_title('Customer Volume by Price')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Conversion by price quartile
    ax4 = axes[1, 1]
    customers_clean['price_quartile'] = pd.qcut(customers_clean[price_var], q=4,
                                                labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
    quartile_conv = customers_clean.groupby('price_quartile')['converted'].mean()
    quartile_counts = customers_clean.groupby('price_quartile').size()

    bars = ax4.bar(range(4), quartile_conv.values, color=['green', 'lightgreen', 'orange', 'red'], alpha=0.7)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(quartile_conv.index)
    ax4.set_ylabel('Customer Conversion Rate')
    ax4.set_title('Conversion Rate by Price Quartile')
    ax4.grid(True, alpha=0.3, axis='y')

    for i, (bar, (idx, count)) in enumerate(zip(bars, quartile_counts.items())):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'n={count:,}\n{height:.1%}', ha='center')

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\n" + "=" * 80)
    print("THRESHOLD EFFECT: KEY FINDINGS")
    print("=" * 80)

    print(f"\nConversion by Price Quartile:")
    for i, (quartile, conv) in enumerate(quartile_conv.items()):
        print(f"  {quartile}: {conv:.1%} (n={quartile_counts.iloc[i]:,})")

    q1_vs_q4 = quartile_conv.iloc[0] - quartile_conv.iloc[-1]
    print(f"\nDifference between lowest and highest quartile: {q1_vs_q4 * 100:.1f} percentage points")

    # Check for sweet spots (where conversion is above average)
    above_avg = binned_stats[binned_stats['conversion_rate'] > customers_clean['converted'].mean()]
    if len(above_avg) > 0:
        print(f"\nSweet spots (above-average conversion):")
        for _, row in above_avg.iterrows():
            print(f"  €{row['avg_price']:,.0f}: {row['conversion_rate']:.1%} (n={row['customer_count']:,})")


def visualize_conversion_by_price_sweet_spots_variant(customers, price_var='max_out_of_pocket'):
    customers_clean = remove_price_outliers(customers, price_var=price_var)
    binned_stats = price_binned_stats(customers_clean, price_var=price_var)

    fig, axes = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle('Customer-Level Price Thresholds: The Two Sweet Spots', fontsize=16, fontweight='bold')

    # ============================================================================
    # PLOT 1: Full curve with highlighted regions
    # ============================================================================
    ax1 = axes
    smoothed = binned_stats['conversion_rate'].rolling(window=5, center=True).mean()
    ax1.plot(binned_stats['avg_price'], smoothed, 'b-', linewidth=3, label='Conversion rate')

    # DYNAMIC: Calculate region boundaries from data
    price_low_end = customers_clean['max_out_of_pocket'].quantile(0.33)  # Bottom 33%
    price_mid_end = customers_clean['max_out_of_pocket'].quantile(0.67)  # Top 33%
    price_high_end = customers_clean['max_out_of_pocket'].max() * 0.8  # 80% of max

    # Use actual data-driven boundaries instead of hardcoded ones
    ax1.axvspan(0, price_low_end, alpha=0.2, color='green',
                label=f'Low-end (€0-{price_low_end:,.0f})')
    ax1.axvspan(price_low_end, price_mid_end, alpha=0.2, color='red',
                label=f'Mid-range (€{price_low_end:,.0f}-{price_mid_end:,.0f})')
    ax1.axvspan(price_mid_end, price_high_end, alpha=0.2, color='gold',
                label=f'High-end (€{price_mid_end:,.0f}+)')

    # Add overall average line (already dynamic)
    overall_avg = customers_clean['converted'].mean() * 100
    ax1.axhline(y=overall_avg / 100, color='gray',
                linestyle='--', alpha=0.7, label=f'Overall avg: {overall_avg:.1f}%')

    ax1.set_xlabel('Maximum Quote Price (€)')
    ax1.set_ylabel('Customer Conversion Rate')
    ax1.set_title('The Two Sweet Spots: Low-End and Heat Pump Territory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()


def report_product_conversion_by_price_quartile(customers_clean):
    # Create price quartiles in the customer dataset
    customers_clean['price_quartile'] = pd.qcut(customers_clean['max_out_of_pocket'],
                                                q=4, labels=['Q1_Low', 'Q2_MidLow', 'Q3_MidHigh', 'Q4_High'])

    # Analyze product mix by quartile
    print("\n" + "=" * 80)
    print("Product mix by price quartile")
    print("=" * 80)

    product_by_quartile = pd.crosstab(
        customers_clean['price_quartile'],
        customers_clean['main_equipment_category'],
        normalize='index'
    ) * 100

    print("\nProduct distribution by price quartile (%):")
    print(product_by_quartile.round(1))

    # Conversion by product and quartile
    print("\n" + "=" * 80)
    print("Conversion rates by product and price quartile")
    print("=" * 80)

    for product in ['Heat Pump', 'Boiler', 'AC', 'Stove']:
        subset = customers_clean[customers_clean['main_equipment_category'] == product]
        if len(subset) > 100:
            print(f"\n{product} (n={len(subset):,}):")
            quartile_conv = subset.groupby('price_quartile')['converted'].mean() * 100
            for quartile, conv in quartile_conv.items():
                print(f"  {quartile}: {conv:.1f}%")

    # Heat pump specific analysis
    hp_customers = customers_clean[customers_clean['ever_bought_heat_pump']]
    print("\n" + "=" * 80)
    print("HEAT PUMP CUSTOMERS (n={:,})".format(len(hp_customers)))
    print("=" * 80)

    print(f"Average price: €{hp_customers['max_out_of_pocket'].mean():,.0f}")
    print(f"Conversion rate: {hp_customers['converted'].mean():.1%}")
    print(f"Subsidy issue rate: {hp_customers['had_subsidy_issue'].mean():.1%}")

    # Where do heat pumps appear?
    hp_by_quartile = customers_clean.groupby('price_quartile')['ever_bought_heat_pump'].mean() * 100
    print("\nHeat pump adoption by quartile:")
    for quartile, pct in hp_by_quartile.items():
        print(f"  {quartile}: {pct:.1f}%")


def visualize_conversion_by_equipment_category_price(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Equipment Category Segment Effect")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)

    # Segment 1: By equipment category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Price-Conversion Curves by Equipment Category (Customer Level)', fontsize=16, fontweight='bold')

    products = ['Stove', 'Heat Pump', 'Boiler', 'AC']
    colors = ['purple', 'green', 'blue', 'orange']

    for idx, (product, color) in enumerate(zip(products, colors)):
        ax = axes[idx // 2, idx % 2]

        subset = customers_clean[customers_clean['main_equipment_category'] == product]
        if len(subset) > 200:
            # Create price bins
            subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=15, duplicates='drop')
            bin_conv = subset.groupby('price_bin')['converted'].mean()
            bin_price = subset.groupby('price_bin')['max_out_of_pocket'].mean()

            ax.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2,
                    label=f'{product} (n={len(subset):,})')
            ax.axhline(y=subset['converted'].mean(), color=color, linestyle='--', alpha=0.5)

        ax.set_xlabel('Maximum Quote Price (€)')
        ax.set_ylabel('Customer Conversion Rate')
        ax.set_title(f'{product} Customers')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def visualise_conversion_by_brand_price(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Conversion by Brand Price")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)

    # Get top 8 brands by volume
    top_brands = customers_clean['main_brand'].value_counts().head(8).index.tolist()
    print(f"Top 8 brands: {top_brands}")

    # Create subplots (2 rows, 4 columns for 8 brands)
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Price-Conversion Curves by Brand (TTC Price)', fontsize=16, fontweight='bold')

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for idx, brand in enumerate(top_brands):
        ax = axes[idx]

        subset = customers_clean[customers_clean['main_brand'] == brand]

        if len(subset) > 100:  # Need enough data
            # Create price bins (using out_of_pocket which is TTC after subsidies)
            subset['price_bin'] = pd.qcut(subset['avg_out_of_pocket'], q=10, duplicates='drop')
            bin_conv = subset.groupby('price_bin')['converted'].mean() * 100
            bin_price = subset.groupby('price_bin')['avg_out_of_pocket'].mean()

            # Plot main curve
            ax.plot(bin_price, bin_conv, 'o-', color='steelblue', linewidth=2, markersize=6)

            # Add horizontal line for brand average
            brand_avg = subset['converted'].mean() * 100
            ax.axhline(y=brand_avg, color='red', linestyle='--', alpha=0.5,
                       label=f'Avg: {brand_avg:.1f}%')

            # Add vertical line for median price
            median_price = subset['avg_out_of_pocket'].median()
            ax.axvline(x=median_price, color='gray', linestyle=':', alpha=0.5,
                       label=f'Median: €{median_price:,.0f}')

            # Add confidence bands (simplified)
            bin_std = subset.groupby('price_bin')['converted'].std() * 100
            ax.fill_between(bin_price,
                            bin_conv - bin_std,
                            bin_conv + bin_std,
                            alpha=0.2, color='steelblue')

            ax.set_xlabel('Price TTC After Subsidies (€)')
            ax.set_ylabel('Conversion Rate (%)')
            ax.set_title(f'{brand}\n(n={len(subset):,})')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 30000)
            ax.set_ylim(0, 70)
        else:
            ax.text(0.5, 0.5, f'{brand}\nInsufficient data\n(n={len(subset)})',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(brand)

    plt.tight_layout()
    plt.show()


def show_two_panel_conversion_by_brand_price(customers, price_var='max_out_of_pocket'):
    customers_clean = remove_price_outliers(customers, price_var=price_var)
    top_brands = customers_clean['main_brand'].value_counts().head(8).index.tolist()
    colors = ['purple', 'green', 'blue', 'orange']

    # Create a 2-panel figure: individual curves + summary table
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Brand Price Analysis: TTC Conversion Curves', fontsize=16, fontweight='bold')

    # Panel 1: Individual brand curves
    ax1 = axes[0]

    for idx, (brand, color) in enumerate(zip(top_brands, colors)):
        subset = customers_clean[customers_clean['main_brand'] == brand]

        if len(subset) > 200:
            subset['price_bin'] = pd.qcut(subset['avg_out_of_pocket'], q=8, duplicates='drop')
            bin_conv = subset.groupby('price_bin')['converted'].mean() * 100
            bin_price = subset.groupby('price_bin')['avg_out_of_pocket'].mean()

            ax1.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2,
                     label=brand, alpha=0.7, markersize=4)

    ax1.set_xlabel('Price TTC After Subsidies (€)')
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Brand Price Sensitivity')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.set_xlim(0, 30000)
    ax1.set_ylim(0, 70)

    # Panel 2: Summary statistics table
    ax2 = axes[1]
    ax2.axis('off')

    # Prepare summary data
    summary_data = []
    for brand in top_brands:
        subset = customers_clean[customers_clean['main_brand'] == brand]
        if len(subset) > 100:
            # Find best price range
            subset['price_bin'] = pd.qcut(subset['avg_out_of_pocket'], q=5, duplicates='drop')
            bin_stats = subset.groupby('price_bin')['converted'].mean()
            best_bin = bin_stats.idxmax()
            best_conv = bin_stats.max() * 100
            best_price_range = f"€{best_bin.left:,.0f}-{best_bin.right:,.0f}"

            summary_data.append([
                brand,
                f"{len(subset):,}",
                f"{subset['converted'].mean() * 100:.1f}%",
                f"€{subset['avg_out_of_pocket'].mean():,.0f}",
                best_price_range,
                f"{best_conv:.1f}%"
            ])

    # Create table
    table = ax2.table(cellText=summary_data,
                      colLabels=['Brand', 'Customers', 'Avg Conv', 'Avg Price', 'Best Price Range', 'Best Conv'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.15, 0.1, 0.1, 0.15, 0.25, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code by performance
    for i, row in enumerate(summary_data):
        conv = float(row[2].strip('%'))
        if conv > 45:
            table[(i + 1, 2)].set_facecolor('lightgreen')
        elif conv < 35:
            table[(i + 1, 2)].set_facecolor('lightcoral')

    ax2.set_title('Brand Performance Summary', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def visualize_conversion_by_equipment_type_brand_price(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("TTC Price-conversion curves: 4 equipment types, multiple brands per graph")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)

    # Equipment types to analyze
    equipment_types = ['Heat Pump', 'Boiler', 'Stove', 'AC']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('TTC Price-Conversion Curves: One Graph Per Equipment Type', fontsize=16, fontweight='bold')

    # Flatten axes for iteration
    axes = axes.flatten()

    for idx, equipment in enumerate(equipment_types):
        ax = axes[idx]

        # Filter for this equipment type
        equip_subset = customers_clean[customers_clean['main_equipment_category'] == equipment]

        # Get top brands for this equipment type (minimum 50 customers)
        brand_counts = equip_subset['main_brand'].value_counts()
        top_brands_equip = brand_counts[brand_counts > 50].head(6).index.tolist()

        print(f"\n{equipment} - Top brands: {top_brands_equip}")

        # Use colormap for brands
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_brands_equip)))

        for brand_idx, (brand, color) in enumerate(zip(top_brands_equip, colors)):
            brand_subset = equip_subset[equip_subset['main_brand'] == brand]

            if len(brand_subset) > 30:  # Need enough data points
                # Create price bins using TTC out_of_pocket
                try:
                    brand_subset['price_bin'] = pd.qcut(brand_subset['avg_out_of_pocket'], q=5, duplicates='drop')
                    bin_conv = brand_subset.groupby('price_bin')['converted'].mean() * 100
                    bin_price = brand_subset.groupby('price_bin')['avg_out_of_pocket'].mean()

                    # Plot line for this brand
                    ax.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2,
                            label=f'{brand} (n={len(brand_subset):,})', alpha=0.8, markersize=4)
                except:
                    print(f"    Skipping {brand} - insufficient price variation")

        # Add average line for this equipment type
        equip_avg = equip_subset['converted'].mean() * 100
        ax.axhline(y=equip_avg, color='black', linestyle='--', linewidth=1,
                   label=f'{equipment} avg: {equip_avg:.1f}%', alpha=0.5)

        # Labels and formatting
        ax.set_xlabel('TTC Price After Subsidies (€)', fontsize=10)
        ax.set_ylabel('Conversion Rate (%)', fontsize=10)
        ax.set_title(f'{equipment} (n={len(equip_subset):,} customers)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7)
        ax.set_xlim(0, 25000)
        ax.set_ylim(0, 80)

    plt.tight_layout()
    plt.show()


def show_conversion_by_price_dashboard(customers, price_var='max_out_of_pocket'):

    customers_clean = remove_price_outliers(customers, price_var=price_var)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Create the master price-conversion dashboard
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Price-Conversion Dependency Dashboard', fontsize=20, fontweight='bold', y=0.98)

    # Grid specification
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # ============================================================================
    # PLOT 1: Main Price-Conversion Curve with Confidence Intervals
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Create price bins
    price_var = 'max_out_of_pocket'
    customers_clean['price_bin'] = pd.qcut(customers_clean[price_var], q=30, duplicates='drop')

    binned_stats = customers_clean.groupby('price_bin').agg({
        'converted': ['mean', 'count', 'sem'],
        price_var: 'mean'
    }).round(4)

    binned_stats.columns = ['conversion_rate', 'customer_count', 'std_error', 'avg_price']
    binned_stats['ci_lower'] = binned_stats['conversion_rate'] - 1.96 * binned_stats['std_error']
    binned_stats['ci_upper'] = binned_stats['conversion_rate'] + 1.96 * binned_stats['std_error']
    binned_stats = binned_stats.reset_index()

    # Main curve
    ax1.plot(binned_stats['avg_price'], binned_stats['conversion_rate'] * 100,
             'b-', linewidth=3, label='Conversion rate')

    # Confidence interval
    ax1.fill_between(binned_stats['avg_price'],
                     binned_stats['ci_lower'] * 100,
                     binned_stats['ci_upper'] * 100,
                     alpha=0.2, color='blue', label='95% CI')

    # Overall average
    overall_conv = customers_clean['converted'].mean() * 100
    ax1.axhline(y=overall_conv, color='red', linestyle='--', linewidth=2,
                label=f'Overall avg: {overall_conv:.1f}%')

    # Highlight sweet spots
    # Use price quantiles to define regions
    q20 = customers_clean['max_out_of_pocket'].quantile(0.20)
    q40 = customers_clean['max_out_of_pocket'].quantile(0.40)
    q60 = customers_clean['max_out_of_pocket'].quantile(0.60)
    q80 = customers_clean['max_out_of_pocket'].quantile(0.80)

    # Define regions based on conversion patterns
    ax1.axvspan(0, q20, alpha=0.2, color='green', label=f'Entry level (0-€{q20:,.0f})')
    ax1.axvspan(q20, q60, alpha=0.2, color='yellow', label=f'Mid-range (€{q20:,.0f}-{q60:,.0f})')
    ax1.axvspan(q60, q80, alpha=0.2, color='orange', label=f'Premium (€{q60:,.0f}-{q80:,.0f})')
    ax1.axvspan(q80, customers_clean['max_out_of_pocket'].max(), alpha=0.2, color='gold',
                label=f'Luxury (€{q80:,.0f}+)')

    ax1.set_xlabel('Price (€)', fontsize=12)
    ax1.set_ylabel('Customer Conversion Rate (%)', fontsize=12)
    ax1.set_title('1. Price-Conversion Curve with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 30000)
    ax1.set_ylim(20, 60)

    # ============================================================================
    # PLOT 2: Price Distribution by Conversion Status
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    converted_prices = customers_clean[customers_clean['converted'] == 1]['max_out_of_pocket']
    nonconverted_prices = customers_clean[customers_clean['converted'] == 0]['max_out_of_pocket']

    ax2.hist([converted_prices, nonconverted_prices], bins=50, density=True,
             alpha=0.6, label=['Converted', 'Not Converted'], color=['green', 'red'])
    ax2.set_xlabel('Price (€)')
    ax2.set_ylabel('Density')
    ax2.set_title('2. Price Distribution: Converted vs Not Converted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30000)

    # ============================================================================
    # PLOT 3: Violin Plot - Price by Conversion
    # ============================================================================
    ax3 = fig.add_subplot(gs[0, 3])

    data_to_plot = [nonconverted_prices, converted_prices]
    parts = ax3.violinplot(data_to_plot, positions=[0, 1],
                           showmedians=True, showextrema=True)
    parts['bodies'][0].set_facecolor('red')
    parts['bodies'][0].set_alpha(0.5)
    parts['bodies'][1].set_facecolor('green')
    parts['bodies'][1].set_alpha(0.5)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Not Converted', 'Converted'])
    ax3.set_ylabel('Price (€)')
    ax3.set_title('3. Price Distribution by Conversion Status')
    ax3.grid(True, alpha=0.3, axis='y')

    # ============================================================================
    # PLOT 4: Price-Conversion by Product Category
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    products = ['Stove', 'Heat Pump', 'Boiler', 'AC']
    colors = ['purple', 'green', 'blue', 'orange']

    for product, color in zip(products, colors):
        subset = customers_clean[customers_clean['main_equipment_category'] == product]
        if len(subset) > 200:
            subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=15, duplicates='drop')
            bin_conv = subset.groupby('price_bin')['converted'].mean() * 100
            bin_price = subset.groupby('price_bin')['max_out_of_pocket'].mean()
            ax4.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2,
                     label=f'{product} (n={len(subset):,})')

    ax4.set_xlabel('Price (€)')
    ax4.set_ylabel('Conversion Rate (%)')
    ax4.set_title('4. Price-Conversion by Product Category')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(0, 30000)

    # ============================================================================
    # PLOT 5: Heatmap - Conversion by Price and Product
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    price_bins = pd.cut(customers_clean['max_out_of_pocket'], bins=8)
    product_bins = customers_clean['main_equipment_category']

    pivot_table = customers_clean.pivot_table(
        values='converted',
        index=product_bins,
        columns=price_bins,
        aggfunc='mean'
    ) * 100

    im = ax5.imshow(pivot_table, cmap='RdYlGn', aspect='auto', vmin=20, vmax=55)
    ax5.set_yticks(range(len(pivot_table.index)))
    ax5.set_yticklabels(pivot_table.index)
    ax5.set_xticks(range(len(pivot_table.columns)))
    ax5.set_xticklabels([f'€{int(b.left):,}' for b in pivot_table.columns], rotation=45, ha='right')
    ax5.set_title('5. Conversion Heatmap: Price × Product')
    plt.colorbar(im, ax=ax5, label='Conversion %')

    # ============================================================================
    # PLOT 6: Price Sensitivity by Product (Bar Chart)
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    sensitivity_data = []
    for product in products:
        subset = customers_clean[customers_clean['main_equipment_category'] == product]
        if len(subset) > 200:
            low_price = subset['max_out_of_pocket'] <= subset['max_out_of_pocket'].quantile(0.25)
            high_price = subset['max_out_of_pocket'] >= subset['max_out_of_pocket'].quantile(0.75)
            low_conv = subset[low_price]['converted'].mean() * 100
            high_conv = subset[high_price]['converted'].mean() * 100
            sensitivity_data.append({
                'product': product,
                'low_conv': low_conv,
                'high_conv': high_conv,
                'drop': low_conv - high_conv
            })

    sensitivity_df = pd.DataFrame(sensitivity_data)
    sensitivity_df = sensitivity_df.sort_values('drop', ascending=True)

    x = range(len(sensitivity_df))
    width = 0.35
    ax6.bar([i - width / 2 for i in x], sensitivity_df['low_conv'], width, label='Low Price', color='green', alpha=0.7)
    ax6.bar([i + width / 2 for i in x], sensitivity_df['high_conv'], width, label='High Price', color='red', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(sensitivity_df['product'])
    ax6.set_ylabel('Conversion Rate (%)')
    ax6.set_title('6. Price Sensitivity: Low vs High Price')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # ============================================================================
    # PLOT 7: Price-Conversion Over Time (Yearly)
    # ============================================================================
    ax7 = fig.add_subplot(gs[1, 3])

    years = ['2023', '2024', '2025']
    colors_year = ['green', 'blue', 'red']

    for year, color in zip(years, colors_year):
        subset = customers_clean[customers_clean['period'] == year]
        if len(subset) > 500:
            subset['price_bin'] = pd.qcut(subset['max_out_of_pocket'], q=10, duplicates='drop')
            bin_conv = subset.groupby('price_bin')['converted'].mean() * 100
            bin_price = subset.groupby('price_bin')['max_out_of_pocket'].mean()
            ax7.plot(bin_price, bin_conv, 'o-', color=color, linewidth=2,
                     label=f'{year} (avg: {subset["converted"].mean() * 100:.1f}%)')

    ax7.set_xlabel('Price (€)')
    ax7.set_ylabel('Conversion Rate (%)')
    ax7.set_title('7. Price-Conversion Curve Over Time')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    ax7.set_xlim(0, 30000)

    # ============================================================================
    # PLOT 8: 2D Density Plot - Price vs Conversion (Jittered)
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 0])

    # Add jitter to conversion for density plot
    jitter = np.random.normal(0, 0.02, size=len(customers_clean))
    y_jittered = customers_clean['converted'] + jitter

    # Create 2D histogram
    hb = ax8.hexbin(customers_clean['max_out_of_pocket'], y_jittered,
                    gridsize=40, cmap='YlOrRd', mincnt=1)
    ax8.set_xlabel('Price (€)')
    ax8.set_ylabel('Conversion (jittered)')
    ax8.set_title('8. Density: Where Are the Customers?')
    plt.colorbar(hb, ax=ax8, label='Number of customers')
    ax8.set_xlim(0, 30000)

    # ============================================================================
    # PLOT 9: Conversion by Price Decile
    # ============================================================================
    ax9 = fig.add_subplot(gs[2, 1])

    customers_clean['decile'] = pd.qcut(customers_clean['max_out_of_pocket'], q=10,
                                        labels=[f'D{i}' for i in range(1, 11)])
    decile_conv = customers_clean.groupby('decile')['converted'].mean() * 100
    decile_price = customers_clean.groupby('decile')['max_out_of_pocket'].mean()

    bars = ax9.bar(range(10), decile_conv.values, color=plt.cm.RdYlGn(decile_conv.values / 100))
    ax9.set_xticks(range(10))
    ax9.set_xticklabels([f'€{int(p):,}' for p in decile_price.values], rotation=45, ha='right')
    ax9.set_ylabel('Conversion Rate (%)')
    ax9.set_title('9. Conversion by Price Decile')
    ax9.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, decile_conv.values)):
        ax9.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1, f'{val:.1f}%', ha='center', fontsize=8)

    # ============================================================================
    # PLOT 10: Subsidy Issues by Price
    # ============================================================================
    ax10 = fig.add_subplot(gs[2, 2])

    # Create bins for subsidy issue rate
    subsidy_by_price = customers_clean.groupby('price_bin').agg({
        'had_subsidy_issue': 'mean',
        'max_out_of_pocket': 'mean'
    }).reset_index()

    ax10.plot(subsidy_by_price['max_out_of_pocket'], subsidy_by_price['had_subsidy_issue'] * 100,
              'ro-', linewidth=2, markersize=4)
    ax10.set_xlabel('Price (€)')
    ax10.set_ylabel('Subsidy Issue Rate (%)')
    ax10.set_title('10. Subsidy Issues Increase with Price')
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim(0, 30000)

    # ============================================================================
    # PLOT 11: Heat Pump Share by Price
    # ============================================================================
    ax11 = fig.add_subplot(gs[2, 3])

    hp_share = customers_clean.groupby('price_bin')['ever_bought_heat_pump'].mean() * 100
    ax11.fill_between(binned_stats['avg_price'], 0, hp_share.values, alpha=0.5, color='green')
    ax11.plot(binned_stats['avg_price'], hp_share.values, 'g-', linewidth=2)
    ax11.set_xlabel('Price (€)')
    ax11.set_ylabel('Heat Pump Share (%)')
    ax11.set_title('11. Heat Pumps Dominate High End')
    ax11.grid(True, alpha=0.3)
    ax11.set_xlim(0, 30000)
    ax11.set_ylim(0, 100)

    # ============================================================================
    # DYNAMIC INSIGHTS CALCULATION
    # ============================================================================

    # 1. Find top 2 sweet spots (highest converting price bins)
    top_bins = binned_stats.nlargest(2, 'conversion_rate')
    sweet1_price = top_bins.iloc[0]['avg_price'] / 1000
    sweet1_conv = top_bins.iloc[0]['conversion_rate'] * 100
    sweet2_price = top_bins.iloc[1]['avg_price'] / 1000
    sweet2_conv = top_bins.iloc[1]['conversion_rate'] * 100

    # 2. Find dead zone (lowest converting bin)
    worst_bin = binned_stats.nsmallest(1, 'conversion_rate').iloc[0]
    dead_low = worst_bin['price_bin'].left / 1000
    dead_high = worst_bin['price_bin'].right / 1000
    dead_conv_low = worst_bin['conversion_rate'] * 100
    dead_conv_high = binned_stats['conversion_rate'].quantile(0.25) * 100

    # 3. Heat pump metrics
    hp_customers = customers_clean[customers_clean['ever_bought_heat_pump'] == True]
    hp_subsidy = hp_customers['had_subsidy_issue'].mean() * 100
    hp_conv = hp_customers['converted'].mean() * 100

    # 4. Price sensitivity from the sensitivity dataframe
    sensitivity_str = " > ".join([
        f"{row['product']} ({row['drop']:.1f}pts)"
        for _, row in sensitivity_df.sort_values('drop', ascending=False).iterrows()
    ])

    # 5. Decision time
    same_day_pct = (customers_clean['decision_days'] == 1).mean() * 100

    # ============================================================================
    # Add dynamic summary annotations
    # ============================================================================
    fig.text(0.02, 0.02,
             f"Key Insights:\n"
             f"• Two sweet spots: €{sweet1_price:.1f}k ({sweet1_conv:.1f}%) and €{sweet2_price:.1f}k ({sweet2_conv:.1f}%)\n"
             f"• Dead zone: €{dead_low:.1f}k-{dead_high:.1f}k ({dead_conv_low:.0f}-{dead_conv_high:.0f}%)\n"
             f"• Heat pumps: {hp_subsidy:.0f}% subsidy issues, {hp_conv:.1f}% conversion\n"
             f"• Price sensitivity: {sensitivity_str}\n"
             f"• {same_day_pct:.0f}% decide in 1 day - price is not the only factor",
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Print dynamic values for verification
    print("\n" + "=" * 80)
    print("DYNAMIC INSIGHTS CALCULATED:")
    print("=" * 80)
    print(f"Sweet spots: €{sweet1_price:.1f}k ({sweet1_conv:.1f}%) and €{sweet2_price:.1f}k ({sweet2_conv:.1f}%)")
    print(f"Dead zone: €{dead_low:.1f}k-{dead_high:.1f}k ({dead_conv_low:.0f}-{dead_conv_high:.0f}%)")
    print(f"Heat pumps: {hp_subsidy:.0f}% subsidy issues, {hp_conv:.1f}% conversion")
    print(f"Price sensitivity: {sensitivity_str}")
    print(f"Same day decisions: {same_day_pct:.0f}%")

    plt.tight_layout()
    plt.savefig('price_conversion_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_product_type_price_conversion_heatmap(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Product Type - Price Conversion Heatmap")
    print("=" * 80)

    try:
        # Create simple price bins
        customers['price_bin_simple'] = pd.qcut(customers['max_out_of_pocket'], q=5, duplicates='drop')

        # Create simple product bins (if available)
        if 'main_equipment_category' in customers.columns:
            # Create pivot
            simple_pivot = customers.pivot_table(
                values='converted',
                index='main_equipment_category',
                columns='price_bin_simple',
                aggfunc='mean'
            ) * 100

            print("Simple pivot table:")
            print(simple_pivot)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(simple_pivot, cmap='RdYlGn', aspect='auto')
            ax.set_yticks(range(len(simple_pivot.index)))
            ax.set_yticklabels(simple_pivot.index)
            ax.set_xticks(range(len(simple_pivot.columns)))
            ax.set_xticklabels([f'{int(b.left):,}-{int(b.right):,}' for b in simple_pivot.columns], rotation=45)
            plt.colorbar(im, ax=ax)
            plt.title('Product Type Price-Conversion Heatmap')
            plt.tight_layout()
            plt.show()
        else:
            print("No product category column found")

    except Exception as e:
        print(f"Error in simple heatmap: {e}")

    # Check if LBC might be in main_product_family
    if 'main_product_family' in customers.columns:
        print("\n" + "=" * 80)
        print("CHECKING MAIN PRODUCT FAMILY")
        print("=" * 80)
        print(customers['main_product_family'].value_counts().head(20))
