import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from etl.price import remove_price_outliers


def visualise_heat_pump_performance(customers, price_var='max_out_of_pocket'):
    print("\n" + "=" * 80)
    print("Heat Pump Performance Over Time")
    print("=" * 80)

    customers_clean = remove_price_outliers(customers, price_var=price_var)

    hp_over_time = customers_clean[customers_clean['ever_bought_heat_pump']].groupby('period').agg({
        'converted': ['mean', 'count'],
        'max_out_of_pocket': 'mean',
        'had_subsidy_issue': 'mean'
    }).round(3)

    hp_over_time.columns = ['conversion_rate', 'customer_count', 'avg_price', 'subsidy_issue_rate']
    print("\nHeat pump customers by year:")
    print(hp_over_time)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Heat Pump Performance During the Crisis', fontsize=14, fontweight='bold')

    # Plot 1: Heat pump conversion vs overall
    ax1 = axes[0]
    overall_conv = customers_clean.groupby('period')['converted'].mean() * 100
    hp_conv = customers_clean[customers_clean['ever_bought_heat_pump'].fillna(False)].groupby('period')[
                  'converted'].mean() * 100

    x = range(len(overall_conv))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], overall_conv.values, width, label='Overall', color='gray', alpha=0.7)
    ax1.bar([i + width / 2 for i in x], hp_conv.values, width, label='Heat Pump', color='green', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(overall_conv.index)
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Heat Pump vs Overall Conversion')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Heat pump volume trend
    ax2 = axes[1]
    hp_volume = customers_clean[customers_clean['ever_bought_heat_pump'].fillna(False)].groupby('period').size()
    total_volume = customers_clean.groupby('period').size()
    hp_share = (hp_volume / total_volume * 100).fillna(0)

    ax2.plot(hp_share.index, hp_share.values, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Heat Pump Share of Customers (%)')
    ax2.set_title('Heat Pump Adoption Trend')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_product_type_by_brand_heatmap(customers):

    # Create Product/Brand Heatmap (Product Type as rows, Brand as columns)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle('Product vs Brand Analysis (Product Types × Brands)', fontsize=16, fontweight='bold')

    # ============================================================================
    # HEAT MAP 1: Volume (Number of Customers) by Product and Brand
    # ============================================================================
    ax1 = axes[0]

    # Get top brands and product categories
    top_brands = customers['main_brand'].value_counts().head(12).index.tolist()
    product_cats = ['Heat Pump', 'Boiler', 'Stove', 'AC', 'Other']

    # Create a crosstab of product vs brand (counts)
    product_brand_counts = pd.crosstab(
        customers['main_equipment_category'],
        customers['main_brand']
    )

    # Filter to top brands only
    product_brand_counts = product_brand_counts[top_brands]

    # Create heatmap (rows = products, columns = brands)
    im1 = ax1.imshow(product_brand_counts, cmap='YlOrRd', aspect='auto')
    ax1.set_yticks(range(len(product_cats)))
    ax1.set_yticklabels(product_cats)
    ax1.set_xticks(range(len(top_brands)))
    ax1.set_xticklabels(top_brands, rotation=45, ha='right')
    ax1.set_xlabel('Brand')
    ax1.set_ylabel('Product Type')
    ax1.set_title('Customer Volume: Number of Customers by Product and Brand')

    # Add colorbar
    plt.colorbar(im1, ax=ax1, label='Number of Customers')

    # Add value annotations
    for i in range(len(product_cats)):
        for j in range(len(top_brands)):
            value = product_brand_counts.iloc[i, j]
            if value > 50:  # Only show larger values
                color = 'white' if value > 500 else 'black'
                ax1.text(j, i, f'{value}', ha='center', va='center', color=color, fontweight='bold')

    # ============================================================================
    # HEAT MAP 2: Conversion Rate by Product and Brand
    # ============================================================================
    ax2 = axes[1]

    # Create pivot table of conversion rate by product and brand
    product_brand_conv = customers.pivot_table(
        values='converted',
        index='main_equipment_category',
        columns='main_brand',
        aggfunc='mean'
    ) * 100

    # Filter to top brands only
    product_brand_conv = product_brand_conv[top_brands]

    # Create heatmap (rows = products, columns = brands)
    im2 = ax2.imshow(product_brand_conv, cmap='RdYlGn', aspect='auto', vmin=20, vmax=60)
    ax2.set_yticks(range(len(product_cats)))
    ax2.set_yticklabels(product_cats)
    ax2.set_xticks(range(len(top_brands)))
    ax2.set_xticklabels(top_brands, rotation=45, ha='right')
    ax2.set_xlabel('Brand')
    ax2.set_ylabel('Product Type')
    ax2.set_title('Conversion Rate (%) by Product and Brand')

    # Add colorbar
    plt.colorbar(im2, ax=ax2, label='Conversion Rate (%)')

    # Add value annotations
    for i in range(len(product_cats)):
        for j in range(len(top_brands)):
            value = product_brand_conv.iloc[i, j]
            if not np.isnan(value):
                color = 'white' if value > 45 else 'black'
                ax2.text(j, i, f'{value:.0f}%', ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Detailed Analysis Tables
    # ============================================================================
    print("\n" + "=" * 80)
    print("PRODUCT × BRAND ANALYSIS")
    print("=" * 80)

    # Table 1: Volume (Customer Count)
    print("\n📊 CUSTOMER VOLUME (Number of Customers):")
    print("-" * 80)
    volume_table = product_brand_counts.copy()
    print(volume_table.to_string())

    # Table 2: Conversion Rates
    print("\n" + "=" * 80)
    print("📈 CONVERSION RATES (%):")
    print("-" * 80)
    conv_table = product_brand_conv.round(1)
    print(conv_table.to_string())

    # Table 3: Best Brand for Each Product
    print("\n" + "=" * 80)
    print("🏆 BEST BRAND BY PRODUCT TYPE")
    print("-" * 80)

    for product in product_cats:
        if product in product_brand_conv.index:
            product_data = product_brand_conv.loc[product].dropna().sort_values(ascending=False)
            if len(product_data) > 0:
                best_brand = product_data.index[0]
                best_conv = product_data.iloc[0]
                volume = product_brand_counts.loc[product, best_brand]

                print(f"\n{product}:")
                print(f"  Best brand: {best_brand}")
                print(f"  Conversion: {best_conv:.1f}%")
                print(f"  Customers: {volume:,}")

                # Show top 3
                print(f"  Top 3 brands:")
                for i in range(min(3, len(product_data))):
                    brand = product_data.index[i]
                    conv = product_data.iloc[i]
                    vol = product_brand_counts.loc[product, brand]
                    print(f"    {i + 1}. {brand}: {conv:.1f}% (n={vol:,})")

    # Table 4: Best Product for Each Brand
    print("\n" + "=" * 80)
    print("🏆 BEST PRODUCT BY BRAND")
    print("-" * 80)

    for brand in top_brands[:8]:  # Top 8 brands
        if brand in product_brand_conv.columns:
            brand_data = product_brand_conv[brand].dropna().sort_values(ascending=False)
            if len(brand_data) > 0:
                best_product = brand_data.index[0]
                best_conv = brand_data.iloc[0]
                volume = product_brand_counts.loc[best_product, brand]

                print(f"\n{brand}:")
                print(f"  Best product: {best_product}")
                print(f"  Conversion: {best_conv:.1f}%")
                print(f"  Customers: {volume:,}")

                # Show all products for this brand
                print(f"  All products:")
                for product in brand_data.index:
                    conv = brand_data[product]
                    vol = product_brand_counts.loc[product, brand]
                    print(f"    {product}: {conv:.1f}% (n={vol:,})")

    # ============================================================================
    # Strategic Insights
    # ============================================================================
    print("\n" + "=" * 80)
    print("💡 STRATEGIC INSIGHTS")
    print("=" * 80)

    # Find where each brand dominates
    print("\nBrand Dominance by Product Category:")
    for product in product_cats:
        if product in product_brand_counts.index:
            # Find brand with highest volume in this product
            top_volume_brand = product_brand_counts.loc[product].idxmax()
            top_volume = product_brand_counts.loc[product].max()
            top_volume_share = (top_volume / product_brand_counts.loc[product].sum()) * 100

            # Find brand with highest conversion in this product
            if product in product_brand_conv.index:
                top_conv_brand = product_brand_conv.loc[product].dropna().idxmax()
                top_conv = product_brand_conv.loc[product].dropna().max()

                print(f"\n{product}:")
                print(f"  Volume leader: {top_volume_brand} ({top_volume:,} customers, {top_volume_share:.1f}% share)")
                print(f"  Conversion leader: {top_conv_brand} ({top_conv:.1f}%)")

    # Check for "LBC" specifically if it exists
    lbc_brands = [b for b in top_brands if 'LBC' in b.upper()]
    if lbc_brands:
        print("\n" + "=" * 80)
        print("🔍 LBC BRAND ANALYSIS")
        print("=" * 80)
        for lbc in lbc_brands:
            print(f"\n{lbc}:")
            if lbc in product_brand_conv.columns:
                lbc_data = product_brand_conv[lbc].dropna().sort_values(ascending=False)
                print(f"  Best product: {lbc_data.index[0]} ({lbc_data.iloc[0]:.1f}%)")
                for product in lbc_data.index:
                    vol = product_brand_counts.loc[product, lbc]
                    print(f"    {product}: {lbc_data[product]:.1f}% (n={vol:,})")


def show_product_type_by_agency_heatmap(customers):
    # Create Product/Agency Heatmap (Product Type × Agency)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Product × Agency Analysis: Performance Across Sales Agencies', fontsize=16, fontweight='bold')

    # Get top agencies by volume
    top_agencies = customers['main_agency'].value_counts().head(15).index.tolist()
    product_cats = ['Heat Pump', 'Boiler', 'Stove', 'AC', 'Other']

    print(f"\nTop 15 agencies by customer volume:")
    for agency in top_agencies:
        count = customers[customers['main_agency'] == agency].shape[0]
        conv = customers[customers['main_agency'] == agency]['converted'].mean() * 100
        print(f"  {agency}: {count:,} customers, {conv:.1f}% conversion")

    # ============================================================================
    # HEAT MAP 1: Volume (Number of Customers) by Product and Agency
    # ============================================================================
    ax1 = axes[0]

    # Create a crosstab of product vs agency (counts)
    product_agency_counts = pd.crosstab(
        customers['main_equipment_category'],
        customers['main_agency']
    )

    # Filter to top agencies only
    product_agency_counts = product_agency_counts[top_agencies]

    # Create heatmap (rows = products, columns = agencies)
    im1 = ax1.imshow(product_agency_counts, cmap='YlOrRd', aspect='auto')
    ax1.set_yticks(range(len(product_cats)))
    ax1.set_yticklabels(product_cats)
    ax1.set_xticks(range(len(top_agencies)))
    ax1.set_xticklabels(top_agencies, rotation=45, ha='right')
    ax1.set_xlabel('Agency')
    ax1.set_ylabel('Product Type')
    ax1.set_title('Customer Volume: Number of Customers by Product and Agency')

    # Add colorbar
    plt.colorbar(im1, ax=ax1, label='Number of Customers')

    # Add value annotations for larger volumes
    for i in range(len(product_cats)):
        for j in range(len(top_agencies)):
            value = product_agency_counts.iloc[i, j]
            if value > 50:  # Only show larger values
                color = 'white' if value > 300 else 'black'
                ax1.text(j, i, f'{value}', ha='center', va='center', color=color, fontweight='bold')

    # ============================================================================
    # HEAT MAP 2: Conversion Rate by Product and Agency
    # ============================================================================
    ax2 = axes[1]

    # Create pivot table of conversion rate by product and agency
    product_agency_conv = customers.pivot_table(
        values='converted',
        index='main_equipment_category',
        columns='main_agency',
        aggfunc='mean'
    ) * 100

    # Filter to top agencies only
    product_agency_conv = product_agency_conv[top_agencies]

    # Create heatmap (rows = products, columns = agencies)
    im2 = ax2.imshow(product_agency_conv, cmap='RdYlGn', aspect='auto', vmin=20, vmax=60)
    ax2.set_yticks(range(len(product_cats)))
    ax2.set_yticklabels(product_cats)
    ax2.set_xticks(range(len(top_agencies)))
    ax2.set_xticklabels(top_agencies, rotation=45, ha='right')
    ax2.set_xlabel('Agency')
    ax2.set_ylabel('Product Type')
    ax2.set_title('Conversion Rate (%) by Product and Agency')

    # Add colorbar
    plt.colorbar(im2, ax=ax2, label='Conversion Rate (%)')

    # Add value annotations
    for i in range(len(product_cats)):
        for j in range(len(top_agencies)):
            value = product_agency_conv.iloc[i, j]
            if not np.isnan(value):
                color = 'white' if value > 45 else 'black'
                ax2.text(j, i, f'{value:.0f}%', ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Detailed Analysis Tables
    # ============================================================================
    print("\n" + "=" * 80)
    print("PRODUCT × AGENCY ANALYSIS")
    print("=" * 80)

    # Table 1: Volume (Customer Count)
    print("\n📊 CUSTOMER VOLUME BY PRODUCT AND AGENCY:")
    print("-" * 100)
    volume_table = product_agency_counts.copy()
    print(volume_table.to_string())

    # Table 2: Conversion Rates
    print("\n" + "=" * 80)
    print("📈 CONVERSION RATES (%) BY PRODUCT AND AGENCY:")
    print("-" * 100)
    conv_table = product_agency_conv.round(1)
    print(conv_table.to_string())

    # Table 3: Best Agency for Each Product
    print("\n" + "=" * 80)
    print("🏆 BEST AGENCY BY PRODUCT TYPE")
    print("-" * 100)

    for product in product_cats:
        if product in product_agency_conv.index:
            product_data = product_agency_conv.loc[product].dropna().sort_values(ascending=False)
            if len(product_data) > 0:
                best_agency = product_data.index[0]
                best_conv = product_data.iloc[0]
                volume = product_agency_counts.loc[product, best_agency]

                print(f"\n{product}:")
                print(f"  Best agency: {best_agency}")
                print(f"  Conversion: {best_conv:.1f}%")
                print(f"  Customers: {volume:,}")

                # Show top 3 agencies
                print(f"  Top 3 agencies:")
                for i in range(min(3, len(product_data))):
                    agency = product_data.index[i]
                    conv = product_data.iloc[i]
                    vol = product_agency_counts.loc[product, agency]
                    print(f"    {i + 1}. {agency}: {conv:.1f}% (n={vol:,})")

    # Table 4: Best Product for Each Agency
    print("\n" + "=" * 80)
    print("🏆 BEST PRODUCT BY AGENCY")
    print("-" * 100)

    for agency in top_agencies[:10]:  # Top 10 agencies
        if agency in product_agency_conv.columns:
            agency_data = product_agency_conv[agency].dropna().sort_values(ascending=False)
            if len(agency_data) > 0:
                best_product = agency_data.index[0]
                best_conv = agency_data.iloc[0]
                volume = product_agency_counts.loc[best_product, agency]

                print(f"\n{agency}:")
                print(f"  Best product: {best_product}")
                print(f"  Conversion: {best_conv:.1f}%")
                print(f"  Customers: {volume:,}")

                # Show all products for this agency
                print(f"  All products:")
                for product in agency_data.index:
                    conv = agency_data[product]
                    vol = product_agency_counts.loc[product, agency]
                    print(f"    {product}: {conv:.1f}% (n={vol:,})")

    # ============================================================================
    # Agency Performance Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("📊 AGENCY PERFORMANCE SUMMARY")
    print("=" * 80)

    agency_summary = customers.groupby('main_agency').agg({
        'converted': ['mean', 'count'],
        'ever_bought_heat_pump': 'mean',
        'had_subsidy_issue': 'mean',
        'avg_out_of_pocket': 'mean'
    }).round(3)

    agency_summary.columns = ['conversion_rate', 'customer_count', 'heat_pump_pct', 'subsidy_issue_pct', 'avg_price']
    agency_summary['conversion_rate'] = agency_summary['conversion_rate'] * 100
    agency_summary['heat_pump_pct'] = agency_summary['heat_pump_pct'] * 100
    agency_summary['subsidy_issue_pct'] = agency_summary['subsidy_issue_pct'] * 100
    agency_summary = agency_summary.sort_values('customer_count', ascending=False).head(15)

    print("\nTop 15 agencies by volume:")
    print(agency_summary.to_string())

    # ============================================================================
    # Strategic Insights
    # ============================================================================
    print("\n" + "=" * 80)
    print("💡 STRATEGIC INSIGHTS")
    print("=" * 80)

    # Find which agency excels at each product
    print("\nAgency Excellence by Product Category:")
    for product in product_cats:
        if product in product_agency_conv.index:
            # Find agency with highest conversion for this product
            if product in product_agency_conv.index:
                top_conv_agency = product_agency_conv.loc[product].dropna().idxmax()
                top_conv = product_agency_conv.loc[product].dropna().max()
                volume = product_agency_counts.loc[product, top_conv_agency]

                print(f"\n{product}:")
                print(f"  Conversion leader: {top_conv_agency} ({top_conv:.1f}%, n={volume:,})")

    # Find agencies with highest heat pump conversion
    print("\n" + "=" * 80)
    print("🔥 HEAT PUMP SPECIALISTS")
    print("=" * 80)

    heat_pump_agencies = product_agency_conv.loc['Heat Pump'].dropna().sort_values(ascending=False).head(5)
    print("\nTop 5 agencies for Heat Pump conversion:")
    for agency, conv in heat_pump_agencies.items():
        volume = product_agency_counts.loc['Heat Pump', agency]
        print(f"  {agency}: {conv:.1f}% (n={volume:,})")


def show_brand_by_product_type_heatmap(customers):
    # Create Product/Brand Heatmap (Product Type as columns, Brand as rows - swapped axes)
    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle('Product vs Brand Analysis (Brands × Product Types)', fontsize=16, fontweight='bold')

    # ============================================================================
    # HEAT MAP 1: Volume (Number of Customers) by Brand and Product
    # ============================================================================
    ax1 = axes[0]

    # Get top brands and product categories
    top_brands = customers['main_brand'].value_counts().head(12).index.tolist()
    product_cats = ['Heat Pump', 'Boiler', 'Stove', 'AC', 'Other']

    # Create a crosstab of brand vs product (counts) - SWAPPED: brands as rows, products as columns
    brand_product_counts = pd.crosstab(
        customers['main_brand'],
        customers['main_equipment_category']
    )

    # Filter to top brands only
    brand_product_counts = brand_product_counts.loc[top_brands]

    # Create heatmap (rows = brands, columns = products)
    im1 = ax1.imshow(brand_product_counts, cmap='YlOrRd', aspect='auto')
    ax1.set_yticks(range(len(top_brands)))
    ax1.set_yticklabels(top_brands, fontsize=9)
    ax1.set_xticks(range(len(product_cats)))
    ax1.set_xticklabels(product_cats, rotation=45, ha='right')
    ax1.set_xlabel('Product Type')
    ax1.set_ylabel('Brand')
    ax1.set_title('Customer Volume: Number of Customers by Brand and Product')

    # Add colorbar
    plt.colorbar(im1, ax=ax1, label='Number of Customers')

    # Add value annotations
    for i in range(len(top_brands)):
        for j in range(len(product_cats)):
            value = brand_product_counts.iloc[i, j]
            if value > 50:  # Only show larger values
                color = 'white' if value > 500 else 'black'
                ax1.text(j, i, f'{value}', ha='center', va='center', color=color, fontweight='bold')

    # ============================================================================
    # HEAT MAP 2: Conversion Rate by Brand and Product
    # ============================================================================
    ax2 = axes[1]

    # Create pivot table of conversion rate by brand and product - SWAPPED: brands as rows, products as columns
    brand_product_conv = customers.pivot_table(
        values='converted',
        index='main_brand',
        columns='main_equipment_category',
        aggfunc='mean'
    ) * 100

    # Filter to top brands only
    brand_product_conv = brand_product_conv.loc[top_brands]

    # Create heatmap (rows = brands, columns = products)
    im2 = ax2.imshow(brand_product_conv, cmap='RdYlGn', aspect='auto', vmin=20, vmax=60)
    ax2.set_yticks(range(len(top_brands)))
    ax2.set_yticklabels(top_brands, fontsize=9)
    ax2.set_xticks(range(len(product_cats)))
    ax2.set_xticklabels(product_cats, rotation=45, ha='right')
    ax2.set_xlabel('Product Type')
    ax2.set_ylabel('Brand')
    ax2.set_title('Conversion Rate (%) by Brand and Product')

    # Add colorbar
    plt.colorbar(im2, ax=ax2, label='Conversion Rate (%)')

    # Add value annotations
    for i in range(len(top_brands)):
        for j in range(len(product_cats)):
            value = brand_product_conv.iloc[i, j]
            if not np.isnan(value):
                color = 'white' if value > 45 else 'black'
                ax2.text(j, i, f'{value:.0f}%', ha='center', va='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Detailed Analysis Tables
    # ============================================================================
    print("\n" + "=" * 80)
    print("BRAND × PRODUCT ANALYSIS (Swapped Axes)")
    print("=" * 80)

    # Table 1: Volume (Customer Count)
    print("\n📊 CUSTOMER VOLUME (Number of Customers):")
    print("-" * 80)
    volume_table = brand_product_counts.copy()
    print(volume_table.to_string())

    # Table 2: Conversion Rates
    print("\n" + "=" * 80)
    print("📈 CONVERSION RATES (%):")
    print("-" * 80)
    conv_table = brand_product_conv.round(1)
    print(conv_table.to_string())

    # Table 3: Best Product for Each Brand
    print("\n" + "=" * 80)
    print("🏆 BEST PRODUCT BY BRAND")
    print("-" * 80)

    for brand in top_brands[:10]:  # Top 10 brands
        if brand in brand_product_conv.index:
            brand_data = brand_product_conv.loc[brand].dropna().sort_values(ascending=False)
            if len(brand_data) > 0:
                best_product = brand_data.index[0]
                best_conv = brand_data.iloc[0]
                volume = brand_product_counts.loc[brand, best_product]

                print(f"\n{brand}:")
                print(f"  Best product: {best_product}")
                print(f"  Conversion: {best_conv:.1f}%")
                print(f"  Customers: {volume:,}")

                # Show top 3 products
                print(f"  Top 3 products:")
                for i in range(min(3, len(brand_data))):
                    product = brand_data.index[i]
                    conv = brand_data.iloc[i]
                    vol = brand_product_counts.loc[brand, product]
                    print(f"    {i + 1}. {product}: {conv:.1f}% (n={vol:,})")

    # Table 4: Best Brand for Each Product
    print("\n" + "=" * 80)
    print("🏆 BEST BRAND BY PRODUCT TYPE")
    print("-" * 80)

    for product in product_cats:
        if product in brand_product_conv.columns:
            product_data = brand_product_conv[product].dropna().sort_values(ascending=False)
            if len(product_data) > 0:
                best_brand = product_data.index[0]
                best_conv = product_data.iloc[0]
                volume = brand_product_counts.loc[best_brand, product]

                print(f"\n{product}:")
                print(f"  Best brand: {best_brand}")
                print(f"  Conversion: {best_conv:.1f}%")
                print(f"  Customers: {volume:,}")

                # Show top 5 brands for this product
                print(f"  Top 5 brands:")
                for i in range(min(5, len(product_data))):
                    brand = product_data.index[i]
                    conv = product_data.iloc[i]
                    vol = brand_product_counts.loc[brand, product]
                    print(f"    {i + 1}. {brand}: {conv:.1f}% (n={vol:,})")

    # ============================================================================
    # Strategic Insights
    # ============================================================================
    print("\n" + "=" * 80)
    print("💡 STRATEGIC INSIGHTS")
    print("=" * 80)

    # Find where each brand excels
    print("\nBrand Performance by Product Category:")
    for brand in top_brands[:8]:
        if brand in brand_product_conv.index:
            brand_data = brand_product_conv.loc[brand].dropna().sort_values(ascending=False)
            if len(brand_data) > 0:
                print(f"\n{brand}:")
                for i in range(min(3, len(brand_data))):
                    product = brand_data.index[i]
                    conv = brand_data.iloc[i]
                    vol = brand_product_counts.loc[brand, product]
                    print(f"  {product}: {conv:.1f}% (n={vol:,})")

    # Find which brand dominates each product category
    print("\n" + "=" * 80)
    print("🏆 MARKET LEADERS BY PRODUCT CATEGORY")
    print("=" * 80)

    for product in product_cats:
        if product in brand_product_counts.columns:
            # Volume leader
            volume_leader = brand_product_counts[product].idxmax()
            volume_leader_count = brand_product_counts.loc[volume_leader, product]
            total_volume = brand_product_counts[product].sum()
            market_share = (volume_leader_count / total_volume) * 100

            # Conversion leader
            if product in brand_product_conv.columns:
                conv_leader_data = brand_product_conv[product].dropna()
                if len(conv_leader_data) > 0:
                    conv_leader = conv_leader_data.idxmax()
                    conv_leader_rate = conv_leader_data.max()

                    print(f"\n{product}:")
                    print(
                        f"  Volume leader: {volume_leader} ({volume_leader_count:,} customers, {market_share:.1f}% share)")
                    print(f"  Conversion leader: {conv_leader} ({conv_leader_rate:.1f}%)")

    # Check for "LBC" brands specifically
    lbc_brands = [b for b in top_brands if 'LBC' in b.upper()]
    if lbc_brands:
        print("\n" + "=" * 80)
        print("🔍 LBC BRAND ANALYSIS")
        print("=" * 80)
        for lbc in lbc_brands:
            print(f"\n{lbc}:")
            if lbc in brand_product_conv.index:
                lbc_data = brand_product_conv.loc[lbc].dropna().sort_values(ascending=False)
                print(f"  Best product: {lbc_data.index[0]} ({lbc_data.iloc[0]:.1f}%)")
                print(f"  All products:")
                for product in lbc_data.index:
                    vol = brand_product_counts.loc[lbc, product]
                    print(f"    {product}: {lbc_data[product]:.1f}% (n={vol:,})")

    return brand_product_counts, brand_product_conv