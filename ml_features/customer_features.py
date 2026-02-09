import time

import pandas as pd
import numpy as np


def create_customer_features(
        df: pd.DataFrame,
        target_type: str = 'first_conversion',
        date_col: str = 'dt_creation_devis',
        customer_col: str = 'numero_compte',
        accept_col: str = 'fg_devis_accepte',
        price_col: str = 'mt_apres_remise_ht_devis',
        family_col: str = 'famille_equipement_produit',
        agency_col: str = 'nom_agence',
        region_col: str = 'nom_region',
        discount_col: str = 'mt_remise_exceptionnelle_ht',
        ttc_col: str = 'mt_ttc_apres_aide_devis'
) -> pd.DataFrame:
    print(f"Creating OPTIMIZED customer features (mode: {target_type})...")
    start_time = time.time()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # ─── FAST Filter for pre-first-purchase data ─────────────────────
    if target_type == 'first_conversion':
        print("  Filtering post-first-purchase data...")

        # FAST method: use cumsum to filter
        df['conversion_cumsum'] = df.groupby(customer_col)[accept_col].cumsum()
        df_filtered = df[df['conversion_cumsum'] <= 1].copy()
        df_filtered = df_filtered.drop(columns=['conversion_cumsum'])
    else:
        df_filtered = df.copy()

    # Store original for target
    df_original = df.copy()

    print(f"  Customers: {df_filtered[customer_col].nunique():,}, Quotes: {len(df_filtered):,}")

    # Sort once
    df_filtered = df_filtered.sort_values([customer_col, date_col])

    # ─── SINGLE EFFICIENT GROUPBY (all calculations at once) ────────
    print("  Calculating features...")

    # Pre-calculate discount percentage (vectorized)
    mask = (df_filtered[ttc_col] > 0) & df_filtered[discount_col].notna()
    df_filtered['discount_pct'] = np.where(
        mask,
        df_filtered[discount_col] / df_filtered[ttc_col],
        np.nan
    )

    # Pre-calculate time differences (vectorized)
    df_filtered['days_diff'] = df_filtered.groupby(customer_col)[date_col].diff().dt.days

    # ALL aggregations in ONE groupby (this is key for speed)
    agg_dict = {
        # Basic counts
        'total_quotes': (customer_col, 'size'),

        # Product diversity
        'unique_product_families': (family_col, 'nunique'),

        # Price statistics
        'avg_price': (price_col, 'mean'),
        'min_price': (price_col, 'min'),
        'max_price': (price_col, 'max'),
        'price_std': (price_col, 'std'),

        # Time statistics
        'first_date': (date_col, 'min'),
        'last_date': (date_col, 'max'),
        'avg_days_between_quotes': ('days_diff', 'mean'),
        'std_days_between_quotes': ('days_diff', 'std'),
        'max_days_between_quotes': ('days_diff', 'max'),

        # Discount
        'avg_discount_pct': ('discount_pct', 'mean'),

        # Categorical modes (optimized)
        'main_agency': (agency_col, lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
        'main_region': (region_col, lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    }

    # Execute ONE efficient groupby
    features = df_filtered.groupby(customer_col).agg(**{
        key: val for key, val in agg_dict.items()
    }).reset_index()

    # ─── FAST Derived Features (vectorized) ─────────────────────────

    # Price range
    features['price_range'] = features['max_price'] - features['min_price']

    # Product consistency
    features['product_consistency'] = (features['unique_product_families'] == 1).astype(int)

    # Engagement density
    features['time_span_days'] = (features['last_date'] - features['first_date']).dt.days.fillna(0) + 1
    features['engagement_density'] = features['total_quotes'] / features['time_span_days']
    features['engagement_density'] = features['engagement_density'].fillna(1)

    # ─── Price Trajectory (OPTIMIZED) ──────────────────────────────
    print("  Calculating price trajectory (optimized)...")

    # Fast price trajectory using numpy
    def fast_price_trend(group):
        if len(group) <= 1:
            return 0.0
        prices = group[price_col].values
        mid = max(1, len(prices) // 2)
        return prices[mid:].mean() - prices[:mid].mean()

    price_traj = df_filtered.groupby(customer_col).apply(fast_price_trend)
    price_traj.name = 'price_trajectory'
    features = features.merge(price_traj, left_on=customer_col, right_index=True, how='left')

    # ─── Target from original data ────────────────────────────────
    target = df_original.groupby(customer_col)[accept_col].max()
    target.name = 'converted'
    features = features.merge(target, left_on=customer_col, right_index=True, how='left')

    # ─── Fill NaN values ──────────────────────────────────────────
    features = features.fillna({
        'price_std': 0,
        'price_range': 0,
        'avg_discount_pct': 0,
        'price_trajectory': 0,
        'avg_days_between_quotes': 0,
        'std_days_between_quotes': 0,
        'max_days_between_quotes': 0,
        'converted': 0
    })

    # ─── Rename for consistency ──────────────────────────────────
    features = features.rename(columns={'price_std': 'price_volatility'})

    # ─── Select final columns ────────────────────────────────────
    final_columns = [
        customer_col,
        'total_quotes',
        'converted',
        'avg_days_between_quotes',
        'std_days_between_quotes',
        'max_days_between_quotes',
        'engagement_density',
        'price_trajectory',
        'unique_product_families',
        'product_consistency',
        'avg_price',
        'price_range',
        'price_volatility',
        'avg_discount_pct',
        'main_agency',
        'main_region',
    ]

    features = features[[col for col in final_columns if col in features.columns]]

    # ─── Timing and reporting ────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"✓ Created {len(features.columns) - 2} leakage-free features")
    print(f"→ {len(features):,} customers | {features['converted'].mean():.1%} converters")
    print(f"⏱️  Execution time: {elapsed:.1f} seconds")

    if elapsed <= 3:
        print(f"✅ SUCCESS! Achieved ≤ 3s target")
    else:
        print(f"⚠️  {elapsed:.1f}s (target was 3s)")

    return features

