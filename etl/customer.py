import numpy as np
import pandas as pd


def aggregate_customer(df):
    """
    Single, comprehensive customer-level aggregation for all analyses
    """
    # First, create all necessary quote-level columns
    df = df.copy()

    # Create positive subsidy amount (MONEY CUSTOMER ACTUALLY RECEIVES)
    df['positive_cee'] = df['mt_prime_cee'].clip(lower=0)
    df['positive_maprimerenov'] = df['mt_prime_maprimerenov'].clip(lower=0)
    df['positive_subsidy'] = df['positive_cee'] + df['positive_maprimerenov']

    # OUT_OF_POCKET: What customer actually pays after subsidies
    df['out_of_pocket'] = df['mt_ttc_avant_aide_devis'] - df['positive_subsidy']

    # Subsidy flags (negative values = adjustments/clawbacks)
    df['subsidy_issue'] = (df['mt_prime_cee'] < 0) | (df['mt_prime_maprimerenov'] < 0)
    df['subsidy_issue_cee'] = df['mt_prime_cee'] < 0
    df['subsidy_issue_maprimerenov'] = df['mt_prime_maprimerenov'] < 0

    # Negative subsidies (adjustments/clawbacks)
    df['negative_cee'] = df['mt_prime_cee'].clip(upper=0).abs()
    df['negative_maprimerenov'] = df['mt_prime_maprimerenov'].clip(upper=0).abs()
    df['total_negative_subsidy'] = df['negative_cee'] + df['negative_maprimerenov']

    # Net subsidy
    df['net_subsidy'] = df['mt_prime_cee'] + df['mt_prime_maprimerenov']

    # Heat pump flag
    df['is_heat_pump'] = df['regroup_famille_equipement_produit_principal'].str.contains('HEAT_PUMP', case=False, na=False)

    def categorize_equipment(row):
        if pd.isna(row['regroup_famille_equipement_produit_principal']):
            return 'Unknown'
        if row['is_heat_pump']:
            return 'Heat Pump'
        if 'BOILER_GAS' in str(row['regroup_famille_equipement_produit_principal']):
            return 'Boiler'
        if 'AIR_CONDITIONER' in str(row['regroup_famille_equipement_produit_principal']):
            return 'AC'
        if 'STOVE' in str(row['regroup_famille_equipement_produit_principal']):
            return 'Stove'
        return 'Other'

    df['equipment_category'] = df.apply(categorize_equipment, axis=1)

    print("Quote-level columns created:")
    print(f"  - out_of_pocket: {df['out_of_pocket'].notna().sum():,} values")
    print(f"  - subsidy_issue: {df['subsidy_issue'].sum():,} issues ({df['subsidy_issue'].mean():.1%})")

    # Group by customer
    customer_data = df.groupby('numero_compte').agg({
        # Conversion
        'fg_devis_accepte': 'max',

        # Price metrics (using mt_apres_remise_ht_devis for quote amounts)
        'mt_apres_remise_ht_devis': ['mean', 'min', 'max', 'std', 'count'],

        # OUT_OF_POCKET - what customer actually pays (CRITICAL FOR PRICE ANALYSIS)
        'out_of_pocket': ['mean', 'min', 'max', 'std'],

        # Price before subsidy
        'mt_ttc_avant_aide_devis': ['mean', 'min', 'max'],

        # Dates
        'dt_creation_devis': ['min', 'max'],

        # Subsidy info
        'subsidy_issue': 'max',
        'subsidy_issue_cee': 'max',
        'subsidy_issue_maprimerenov': 'max',
        'positive_subsidy': 'sum',
        'total_negative_subsidy': 'sum',
        'net_subsidy': 'sum',
        'mt_prime_cee': 'sum',
        'mt_prime_maprimerenov': 'sum',

        # Product info - take mode (most common)
        'equipment_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'is_heat_pump': 'max',
        'famille_equipement_produit_principal': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',

        # ===== NEW: BRAND INFO =====
        'marque_produit': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',

        # Geography
        'nom_region': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'nom_agence': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',

        # Customer type
        'statut_client': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    })

    # Flatten column names
    customer_data.columns = [
        # From mt_apres_remise_ht_devis
        'converted',
        'avg_quote_amount', 'min_quote_amount', 'max_quote_amount',
        'std_quote_amount', 'total_quotes',

        # FROM OUT_OF_POCKET - CRITICAL FOR PRICE ANALYSIS
        'avg_out_of_pocket', 'min_out_of_pocket', 'max_out_of_pocket', 'std_out_of_pocket',

        # From mt_ttc_avant_aide_devis
        'avg_price_before_subsidy', 'min_price_before_subsidy', 'max_price_before_subsidy',

        # Dates
        'first_quote_date', 'last_quote_date',

        # Subsidy flags
        'had_subsidy_issue', 'had_negative_cee', 'had_negative_maprimerenov',
        'total_positive_subsidy', 'total_negative_subsidy', 'net_subsidy',
        'total_cee', 'total_maprimerenov',

        # Product info
        'main_equipment_category', 'ever_bought_heat_pump', 'main_product_family',

        # ===== NEW: BRAND =====
        'main_brand',

        # Geography
        'main_region', 'main_agency',

        # Customer type
        'customer_type'
    ]

    # Reset index to make numero_compte a column
    customer_data = customer_data.reset_index()

    # =========================================================================
    # DERIVED FEATURES - ALL THE COLUMNS WE NEED FOR ANALYSIS
    # =========================================================================

    # 1. TIME-BASED METRICS
    customer_data['customer_duration_days'] = (
                                                      pd.to_datetime(customer_data['last_quote_date']) -
                                                      pd.to_datetime(customer_data['first_quote_date'])
                                              ).dt.days + 1

    # DECISION DAYS - same as duration (for administrative burden)
    customer_data['decision_days'] = customer_data['customer_duration_days']

    # 2. PRICE METRICS
    # Price range using out_of_pocket (what customer actually pays)
    customer_data['price_range'] = customer_data['max_out_of_pocket'] - customer_data['min_out_of_pocket']

    # Price range using quote amounts (for comparison)
    customer_data['price_range_quotes'] = customer_data['max_quote_amount'] - customer_data['min_quote_amount']

    # Price volatility (standard deviation relative to mean)
    customer_data['price_volatility'] = customer_data['std_out_of_pocket'] / customer_data['avg_out_of_pocket'].replace(
        0, np.nan)

    # PRICE CV - coefficient of variation (for administrative burden)
    customer_data['price_cv'] = customer_data['std_out_of_pocket'] / customer_data['avg_out_of_pocket'].replace(0,
                                                                                                                np.nan)

    # 3. QUOTE METRICS
    # QUOTE COUNT - alias for total_quotes
    customer_data['quote_count'] = customer_data['total_quotes']

    # Flag for multi-quote customers
    customer_data['multiple_quotes'] = (customer_data['total_quotes'] > 1).astype(int)

    # Quote pattern category
    def categorize_quote_pattern(row):
        if row['total_quotes'] == 1:
            return 'Single quote'
        elif row['total_quotes'] == 2:
            return 'Two quotes'
        else:
            return 'Three+ quotes'

    customer_data['quote_pattern'] = customer_data.apply(categorize_quote_pattern, axis=1)

    # 4. DECISION SPEED CATEGORY
    def categorize_duration(days):
        if days == 1:
            return 'Same day decision'
        elif days <= 7:
            return 'Quick (2-7 days)'
        elif days <= 30:
            return 'Medium (8-30 days)'
        else:
            return 'Long (>30 days)'

    customer_data['decision_speed'] = customer_data['customer_duration_days'].apply(categorize_duration)

    # 5. SUBSIDY ISSUE TYPE
    def categorize_subsidy_issues(row):
        if row['had_negative_cee'] and row['had_negative_maprimerenov']:
            return 'Both'
        elif row['had_negative_cee']:
            return 'CEE Only'
        elif row['had_negative_maprimerenov']:
            return 'MaPrimeRénov Only'
        else:
            return 'No Issues'

    customer_data['subsidy_issue_type'] = customer_data.apply(categorize_subsidy_issues, axis=1)

    # 6. TIME-BASED FEATURES FOR SEASONAL ANALYSIS
    customer_data['first_quote_year'] = pd.to_datetime(customer_data['first_quote_date']).dt.year
    customer_data['first_quote_month'] = pd.to_datetime(customer_data['first_quote_date']).dt.month
    customer_data['first_quote_quarter'] = pd.to_datetime(customer_data['first_quote_date']).dt.quarter
    customer_data['year'] = customer_data['first_quote_year']
    customer_data['month'] = customer_data['first_quote_month']
    customer_data['quarter'] = customer_data['first_quote_quarter']

    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    customer_data['season'] = customer_data['first_quote_month'].apply(get_season)

    # Period for year-over-year comparison
    customer_data['period'] = customer_data['first_quote_year'].astype(str)

    # Year-quarter for trends
    customer_data['year_quarter'] = customer_data['first_quote_year'].astype(str) + 'Q' + customer_data[
        'first_quote_quarter'].astype(str)

    # 7. SUSPENSION FLAG (based on French subsidy suspension dates)
    suspension_start = pd.to_datetime('2025-07-01')
    suspension_end = pd.to_datetime('2025-09-30')
    suspension_start2 = pd.to_datetime('2026-01-01')
    suspension_end2 = pd.to_datetime('2026-01-22')

    customer_data['during_suspension'] = (
                                                 (pd.to_datetime(
                                                     customer_data['first_quote_date']) >= suspension_start) &
                                                 (pd.to_datetime(customer_data['first_quote_date']) <= suspension_end)
                                         ) | (
                                                 (pd.to_datetime(
                                                     customer_data['first_quote_date']) >= suspension_start2) &
                                                 (pd.to_datetime(customer_data['first_quote_date']) <= suspension_end2)
                                         )

    print(f"\n✅ Customer dataset created: {len(customer_data):,} customers")
    print(f"✅ Total columns: {len(customer_data.columns)}")
    print(f"✅ Key columns now available:")
    print(f"   - decision_days: from customer_duration_days")
    print(f"   - price_cv: from std_out_of_pocket / avg_out_of_pocket")  # Price Coefficient of Variation
    print(f"   - quote_count: alias for total_quotes")
    print(f"   - main_brand: most common brand per customer")
    print(f"   - price_range: {customer_data['price_range'].mean():.0f} avg")
    print(f"   - price_volatility: {customer_data['price_volatility'].mean():.2f} avg")
    print(customer_data['subsidy_issue_type'].value_counts())

    # Save
    customer_data.to_csv('customer_master_data.csv', index=False)
    print("\n✅ Saved to 'customer_master_data.csv'")

    return customer_data