import pandas as pd
import numpy as np


def create_customer_features(df):
    print("Creating enhanced customer features...")
    print(f"  Total customers: {df['numero_compte'].nunique():,}")
    # Sort by customer and date
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    customer_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        # Basic info
        features['total_quotes'] = len(customer_data)
        features['converted'] = customer_data['fg_devis_accepte'].max()  # Target

        # Temporal patterns (NO FUTURE LEAKAGE)
        if len(customer_data) > 1:
            # Time between quotes patterns
            time_diffs = customer_data['dt_creation_devis'].diff().dt.days.dropna()
            features['avg_days_between_quotes'] = time_diffs.mean() if len(time_diffs) > 0 else 0
            features['std_days_between_quotes'] = time_diffs.std() if len(time_diffs) > 0 else 0
            features['max_days_between_quotes'] = time_diffs.max() if len(time_diffs) > 0 else 0

            # Engagement pattern: early vs late activity
            time_span = (customer_data['dt_creation_devis'].max() - customer_data['dt_creation_devis'].min()).days + 1
            features['engagement_density'] = len(customer_data) / time_span if time_span > 0 else 0

            # Price trajectory (first half vs second half)
            mid_point = len(customer_data) // 2
            first_half_avg = customer_data.iloc[:mid_point]['mt_apres_remise_ht_devis'].mean()
            second_half_avg = customer_data.iloc[mid_point:]['mt_apres_remise_ht_devis'].mean()
            features['price_trajectory'] = second_half_avg - first_half_avg if mid_point > 0 else 0
        else:
            # Single quote customers
            features.update({
                'avg_days_between_quotes': 0,
                'std_days_between_quotes': 0,
                'max_days_between_quotes': 0,
                'engagement_density': 1,  # All quotes in one day
                'price_trajectory': 0
            })

        # Product engagement patterns
        features['unique_product_families'] = customer_data['famille_equipement_produit'].nunique()
        features['product_consistency'] = 1 if features['unique_product_families'] == 1 else 0

        # Price patterns
        features['avg_price'] = customer_data['mt_apres_remise_ht_devis'].mean()
        features['price_range'] = customer_data['mt_apres_remise_ht_devis'].max() - customer_data[
            'mt_apres_remise_ht_devis'].min()
        features['price_volatility'] = customer_data['mt_apres_remise_ht_devis'].std() if len(customer_data) > 1 else 0

        # Agency and region (most common)
        features['main_agency'] = customer_data['nom_agence'].mode()[0] if len(
            customer_data['nom_agence'].mode()) > 0 else 'missing'
        features['main_region'] = customer_data['nom_region'].mode()[0] if len(
            customer_data['nom_region'].mode()) > 0 else 'missing'

        # Discount patterns
        features['avg_discount_pct'] = (
                customer_data['mt_remise_exceptionnelle_ht'] /
                customer_data['mt_ttc_apres_aide_devis']
        ).replace([np.inf, -np.inf], 0).fillna(0).mean()

        customer_features.append(features)

    df_customers = pd.DataFrame(customer_features)
    print(f"✓ Created features for {len(df_customers):,} customers")
    print(f"✓ New features: {list(df_customers.columns)[:10]}...")
    return df_customers