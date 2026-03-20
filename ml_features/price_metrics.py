import numpy as np
import pandas as pd


def create_quote_price_metrics(df):

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
    df['is_heat_pump'] = df['famille_equipement_produit'].str.contains('PAC|POMPE|HEAT|POMPE À CHALEUR', case=False,
                                                                       na=False) | \
                         df['type_equipement_produit'].str.contains('PAC|POMPE|HEAT|POMPE À CHALEUR', case=False,
                                                                    na=False)

    print("Quote-level columns created:")
    print(f"  - out_of_pocket: {df['out_of_pocket'].notna().sum():,} values")
    print(f"  - subsidy_issue: {df['subsidy_issue'].sum():,} issues ({df['subsidy_issue'].mean():.1%})")

    return df


def create_customer_price_metrics(customer_data):



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
    print(f"   - price_range: {customer_data['price_range'].mean():.0f} avg")
    print(f"   - price_volatility: {customer_data['price_volatility'].mean():.2f} avg")

    print(customer_data['subsidy_issue_type'].value_counts())