import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency


def report_customer_conversion_by_subsidy_issue_status(customers):
    # Customer conversion by subsidy issue status
    print("\nCustomer conversion by subsidy issue status:\n")
    issue_conv = customers.groupby('had_subsidy_issue')['converted'].agg(['mean', 'count', 'sum'])
    issue_conv.columns = ['conversion_rate', 'customer_count', 'converted_count']
    print(issue_conv)

    # Statistical test
    contingency = pd.crosstab(customers['had_subsidy_issue'], customers['converted'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square p-value: {p_value:.6e}")
    print(f"Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")

    # Calculate the difference
    conv_no_issue = customers[~customers['had_subsidy_issue']]['converted'].mean()
    conv_with_issue = customers[customers['had_subsidy_issue']]['converted'].mean()
    diff = (conv_with_issue - conv_no_issue) * 100

    print(f"\nConversion without issues: {conv_no_issue:.2%}")
    print(f"Conversion with issues: {conv_with_issue:.2%}")
    print(f"Difference: {diff:.1f} percentage points")
    print(f"Customer-level finding: +{diff:.1f} points")

    return conv_no_issue, conv_with_issue


def report_customer_metrics_by_subsidy_issue_status(customers):
    print("\n" + "=" * 80)
    print("Customer characteristics by subsidy issue status")
    print("=" * 80)

    # Compare key metrics
    metrics = ['total_quotes', 'avg_quote_amount', 'max_quote_amount',
               'customer_duration_days', 'price_range', 'multiple_quotes']

    print("\nMetric comparison:")
    for metric in metrics:
        if metric in customers.columns:
            no_issue_mean = customers[~customers['had_subsidy_issue']][metric].mean()
            issue_mean = customers[customers['had_subsidy_issue']][metric].mean()
            ratio = issue_mean / no_issue_mean if no_issue_mean != 0 else 0
            print(f"\n{metric}:")
            print(f"  No issues: {no_issue_mean:.2f}")
            print(f"  With issues: {issue_mean:.2f}")
            print(f"  Ratio: {ratio:.2f}x")

    # Add categorical analysis
    print("\nQuote pattern distribution by subsidy issue:")
    quote_pattern_pivot = pd.crosstab(
        customers['quote_pattern'],
        customers['had_subsidy_issue'],
        normalize='columns'
    )
    print(quote_pattern_pivot.round(3))

    print("\nDecision speed by subsidy issue:")
    decision_speed_pivot = pd.crosstab(
        customers['decision_speed'],
        customers['had_subsidy_issue'],
        normalize='columns'
    )
    print(decision_speed_pivot.round(3))

    # Product family distribution
    print("\nProduct family by subsidy issue status:")
    product_pivot = pd.crosstab(customers['main_product_family'],
                                customers['had_subsidy_issue'],
                                normalize='columns')
    print(product_pivot.round(3))

    # Conversion by subsidy issue type
    print("\nConversion by subsidy issue type:")
    type_conv = customers.groupby('subsidy_issue_type')['converted'].agg(['mean', 'count'])
    print(type_conv.sort_values('mean', ascending=False))


def visualize_conversion_by_subsidy_issues(customers, conv_no_issue, conv_with_issue):
    # VISUALIZATION
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Customer-Level Analysis: Subsidy Issues and Conversion', fontsize=16, fontweight='bold')

    # Plot 1: Conversion by subsidy issue status
    ax1 = axes[0, 0]
    bars = ax1.bar(['No Subsidy Issues', 'Has Subsidy Issues'],
                   [conv_no_issue, conv_with_issue],
                   color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Customer Conversion Rate')
    ax1.set_title('Customer Conversion: With vs Without Subsidy Issues')
    ax1.grid(True, alpha=0.3)

    # Add counts
    for i, (bar, (issue, count)) in enumerate(zip(bars, customers.groupby('had_subsidy_issue').size().items())):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'n={count:,}\n{height:.1%}', ha='center', va='bottom')

    # Plot 2: Conversion by subsidy issue type
    ax2 = axes[0, 1]
    type_conv_mean = customers.groupby('subsidy_issue_type')['converted'].mean().sort_values()
    type_counts = customers['subsidy_issue_type'].value_counts()
    colors = {'No Issues': 'blue', 'CEE Only': 'orange', 'MaPrimeRénov Only': 'purple', 'Both': 'red'}
    bar_colors = [colors.get(x, 'gray') for x in type_conv_mean.index]

    bars = ax2.bar(range(len(type_conv_mean)), type_conv_mean.values, color=bar_colors, alpha=0.7)
    ax2.set_xticks(range(len(type_conv_mean)))
    ax2.set_xticklabels(type_conv_mean.index, rotation=45, ha='right')
    ax2.set_ylabel('Customer Conversion Rate')
    ax2.set_title('Conversion by Type of Subsidy Issue')
    ax2.grid(True, alpha=0.3)

    for i, (idx, val) in enumerate(type_conv_mean.items()):
        ax2.text(i, val + 0.01, f'n={type_counts[idx]:,}', ha='center')

    # Plot 3: Quote volume comparison
    ax3 = axes[1, 0]
    quote_data = []
    for issue in [False, True]:
        subset = customers[customers['had_subsidy_issue'] == issue]
        quote_data.append({
            'Subsidy Issues': 'Yes' if issue else 'No',
            'Avg Quotes': subset['total_quotes'].mean(),
            'Median Quotes': subset['total_quotes'].median()
        })
    quote_df = pd.DataFrame(quote_data)

    x = range(len(quote_df))
    ax3.bar([i - 0.2 for i in x], quote_df['Avg Quotes'], width=0.4, label='Average', alpha=0.7, color='steelblue')
    ax3.bar([i + 0.2 for i in x], quote_df['Median Quotes'], width=0.4, label='Median', alpha=0.7, color='lightblue')
    ax3.set_xticks(x)
    ax3.set_xticklabels(quote_df['Subsidy Issues'])
    ax3.set_ylabel('Number of Quotes')
    ax3.set_title('Quote Volume: With vs Without Subsidy Issues')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Price comparison
    ax4 = axes[1, 1]
    price_data = []
    for issue in [False, True]:
        subset = customers[customers['had_subsidy_issue'] == issue]
        price_data.append({
            'Subsidy Issues': 'Yes' if issue else 'No',
            'Avg Price': subset['avg_quote_amount'].mean(),
            'Median Price': subset['avg_quote_amount'].median()
        })
    price_df = pd.DataFrame(price_data)

    x = range(len(price_df))
    ax4.bar([i - 0.2 for i in x], price_df['Avg Price'], width=0.4, label='Average', alpha=0.7, color='steelblue')
    ax4.bar([i + 0.2 for i in x], price_df['Median Price'], width=0.4, label='Median', alpha=0.7, color='lightblue')
    ax4.set_xticks(x)
    ax4.set_xticklabels(price_df['Subsidy Issues'])
    ax4.set_ylabel('Average Quote Amount (€)')
    ax4.set_title('Quote Value: With vs Without Subsidy Issues')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def report_customer_subsidy_issues_by_product(customers):
    # First, let's see what values actually exist in your data
    print("Unique values in main_product_family:")
    print(customers['main_product_family'].value_counts())

    # Get the top product categories from the actual data
    product_counts = customers['main_product_family'].value_counts()
    print("\nTop products:")
    print(product_counts.head(10))

    # Use the actual top products from your data
    # Let's take top 4 for clarity
    top_products = product_counts.head(4).index.tolist()
    print(f"\nUsing these products: {top_products}")

    # Calculate percentages for customers WITHOUT subsidy issues
    no_issue_customers = customers[~customers['had_subsidy_issue']]
    no_issue_pct = []
    for product in top_products:
        pct = (no_issue_customers['main_product_family'] == product).mean() * 100
        no_issue_pct.append(pct)

    # Calculate percentages for customers WITH subsidy issues
    issue_customers = customers[customers['had_subsidy_issue']]
    issue_pct = []
    for product in top_products:
        pct = (issue_customers['main_product_family'] == product).mean() * 100
        issue_pct.append(pct)

    print("\nNo Issue %:", no_issue_pct)
    print("Has Issue %:", issue_pct)

    return top_products, no_issue_pct, issue_pct


def visualize_customer_subsidy_issues_by_product(customers, top_products, no_issue_pct, issue_pct):
    # Create figure with only 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Subsidy Issues: Customer-Level Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Product mix (DYNAMIC) - USING ACTUAL DATA VALUES
    ax1 = axes[0]

    x = range(len(top_products))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], no_issue_pct, width, label='No Issues', color='#457b9d')
    ax1.bar([i + width / 2 for i in x], issue_pct, width, label='Has Issues', color='#e63946')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_products, rotation=45, ha='right')
    ax1.set_ylabel('Percentage of Customers (%)')
    ax1.set_title('Product Mix by Subsidy Issue Status')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Average quote value (DYNAMIC)
    ax2 = axes[1]

    # Calculate average quote values dynamically
    avg_no_issue = customers[~customers['had_subsidy_issue']]['avg_quote_amount'].mean()
    avg_with_issue = customers[customers['had_subsidy_issue']]['avg_quote_amount'].mean()
    value_diff = (avg_with_issue / avg_no_issue - 1) * 100

    bars = ax2.bar(['No Issues', 'Has Issues'],
                   [avg_no_issue, avg_with_issue],
                   color=['#457b9d', '#e63946'], alpha=0.8)
    ax2.set_ylabel('Average Quote Value (€)')
    ax2.set_title(f'Issues = {value_diff:.0f}% Higher Value Deals')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, [avg_no_issue, avg_with_issue]):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 200,
                 f'€{val:,.0f}', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()