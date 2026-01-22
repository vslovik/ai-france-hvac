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


def create_customer_features_(df):
    print("Creating enhanced customer features (vectorized)...")
    print(f" Total customers: {df['numero_compte'].nunique():,}")

    # Sort once
    df = df.sort_values(['numero_compte', 'dt_creation_devis'])

    # Pre-compute helpers outside groups
    df['days_since_prev'] = df.groupby('numero_compte')['dt_creation_devis'].diff().dt.days
    df['discount_pct'] = (
        df['mt_remise_exceptionnelle_ht']
        / df['mt_ttc_apres_aide_devis'].replace(0, np.nan)
    ).fillna(0).replace([np.inf, -np.inf], 0)

    # Aggregations with namedagg for clarity
    agg_funcs = {
        'total_quotes': pd.NamedAgg(column='numero_compte', aggfunc='size'),
        'converted': pd.NamedAgg(column='fg_devis_accepte', aggfunc='max'),
        'avg_days_between_quotes': pd.NamedAgg(column='days_since_prev', aggfunc='mean'),
        'std_days_between_quotes': pd.NamedAgg(column='days_since_prev', aggfunc='std'),
        'max_days_between_quotes': pd.NamedAgg(column='days_since_prev', aggfunc='max'),
        'avg_price': pd.NamedAgg(column='mt_apres_remise_ht_devis', aggfunc='mean'),
        'price_volatility': pd.NamedAgg(column='mt_apres_remise_ht_devis', aggfunc='std'),
        'avg_discount_pct': pd.NamedAgg(column='discount_pct', aggfunc='mean'),
        'unique_product_families': pd.NamedAgg(column='famille_equipement_produit', aggfunc='nunique'),
    }

    customer_df = df.groupby('numero_compte').agg(**agg_funcs).reset_index()

    # Custom aggs that need lambda
    customer_df['price_range'] = df.groupby('numero_compte')['mt_apres_remise_ht_devis'].max() - df.groupby('numero_compte')['mt_apres_remise_ht_devis'].min()

    def get_mode(s):
        vc = s.value_counts()
        return vc.index[0] if not vc.empty else 'missing'

    customer_df['main_agency'] = df.groupby('numero_compte')['nom_agence'].apply(get_mode)
    customer_df['main_region'] = df.groupby('numero_compte')['nom_region'].apply(get_mode)

    # Engagement density
    date_min = df.groupby('numero_compte')['dt_creation_devis'].min()
    date_max = df.groupby('numero_compte')['dt_creation_devis'].max()
    time_span = (date_max - date_min).dt.days + 1
    customer_df['engagement_density'] = customer_df['total_quotes'] / time_span.replace(0, 1)
    customer_df['engagement_density'] = customer_df['engagement_density'].fillna(1)

    # Product consistency
    customer_df['product_consistency'] = (customer_df['unique_product_families'] == 1).astype(int)

    # Price trajectory (vectorized approximation: sort index ensures order)
    def price_trajectory(g):
        if len(g) <= 1:
            return 0
        mid = len(g) // 2
        first_half = g['mt_apres_remise_ht_devis'].values[:mid].mean()
        second_half = g['mt_apres_remise_ht_devis'].values[mid:].mean()
        return second_half - first_half

    # Since df is sorted, we can group and apply on the sorted df
    traj = df.groupby('numero_compte', sort=False).apply(price_trajectory, include_groups=False)
    customer_df['price_trajectory'] = traj.values

    # Fill NaNs for single-quote cases
    fill_cols = [
        'avg_days_between_quotes', 'std_days_between_quotes', 'max_days_between_quotes',
        'price_volatility', 'price_range', 'price_trajectory'
    ]
    customer_df[fill_cols] = customer_df[fill_cols].fillna(0)

    print(f"✓ Created features for {len(customer_df):,} customers")
    print(f"✓ New features: {list(customer_df.columns)[:10]}...")

    return customer_df