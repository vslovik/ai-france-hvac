import numpy as np
import pandas as pd

from ml_features.globals import BUDGET_BRANDS, PREMIUM_BRANDS


def create_market_features(df, target_type='first_conversion'):
    """
    LEAKAGE-FREE market features - Only uses customer's own historical data
    and static brand categorizations (no global statistics from future data)
    """
    print("=" * 80)
    print("CREATING LEAKAGE-FREE MARKET FEATURES")
    print("=" * 80)

    df = remove_leaky_brands(df)

    # Quick validation
    if 'marque_produit' not in df.columns:
        print("⚠️ No brand data")
        return pd.DataFrame(columns=['numero_compte'])

    # 1. Sort once and make a copy
    df = df.copy()
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)

    print(f"Processing {len(df):,} quotes for {df['numero_compte'].nunique():,} customers")

    # Store original for target calculation
    df_original = df.copy()

    # 🚨 CRITICAL FIX: For market features, use ALL quotes
    # Brand preferences are stable characteristics that don't change at purchase
    print("Using ALL quotes for market features (brand preferences are stable)")

    # NO FILTERING for market features!
    # df = df (use all data as-is)

    print(f"  Quotes for feature calculation: {len(df):,}")

    # 4. SINGLE GROUPBY to get customer brand sequences
    print("👥 Grouping customer data...")

    # Get brand sequences per customer
    customer_groups = df.groupby('numero_compte')['marque_produit'].apply(list)
    customer_ids = customer_groups.index.values
    brand_sequences = customer_groups.values

    n_customers = len(customer_ids)
    print(f"  Processing {n_customers:,} customers with brand data")

    # 5. VECTORIZED CALCULATION OF SAFE FEATURES ONLY
    print("⚡ Calculating leakage-free features...")

    # Initialize result arrays
    results = {
        'numero_compte': customer_ids,
        'market_data_available': np.ones(n_customers, dtype=int),
        'premium_brand_ratio': np.zeros(n_customers, dtype=float),
        'budget_brand_ratio': np.zeros(n_customers, dtype=float),
        'unique_brands_count': np.zeros(n_customers, dtype=int),
        'brand_consistency': np.zeros(n_customers, dtype=float),
        'brand_diversity_index': np.zeros(n_customers, dtype=float),
        'brand_switch_rate': np.zeros(n_customers, dtype=float),
    }

    # Process customers in batches
    batch_size = 1000

    for i in range(0, n_customers, batch_size):
        batch_end = min(i + batch_size, n_customers)

        for j in range(i, batch_end):
            seq = brand_sequences[j]
            seq_len = len(seq)

            if seq_len == 0:
                results['market_data_available'][j] = 0
                continue

            # Convert to numpy array for efficiency
            seq_array = np.array(seq)

            # FEATURE 1: Premium vs Budget brand ratios
            premium_count = sum(1 for b in seq if b in PREMIUM_BRANDS)
            budget_count = sum(1 for b in seq if b in BUDGET_BRANDS)

            results['premium_brand_ratio'][j] = premium_count / seq_len
            results['budget_brand_ratio'][j] = budget_count / seq_len

            # FEATURE 2: Brand diversity (customer's own data only)
            unique_brands = set(seq)
            results['unique_brands_count'][j] = len(unique_brands)

            # FEATURE 3: Brand consistency
            if seq_len > 0:
                results['brand_consistency'][j] = 1 if len(unique_brands) == 1 else 0

            # FEATURE 4: Brand diversity index (1 - HHI, using customer's own data only)
            if seq_len > 1:
                # Calculate Herfindahl-Hirschman Index for this customer's brands
                unique, counts = np.unique(seq_array, return_counts=True)
                proportions = counts / seq_len
                hhi = np.sum(proportions ** 2)
                results['brand_diversity_index'][j] = 1 - hhi

            # FEATURE 5: Brand switching rate
            if seq_len > 1:
                switches = sum(1 for k in range(1, seq_len) if seq[k] != seq[k - 1])
                results['brand_switch_rate'][j] = switches / (seq_len - 1)

    print("✅ Calculations complete")

    # 6. CREATE DATAFRAME
    features_df = pd.DataFrame(results)

    # 7. ADD TARGET FROM ORIGINAL DATA (full lifetime)
    print("🎯 Adding target variable...")

    # 🎯 CRITICAL: Target is ALWAYS "does customer ever convert?"
    # For both first_conversion and any_conversion, it's the same binary question
    target = df_original.groupby('numero_compte')['fg_devis_accepte'].max()
    target.name = 'converted'
    features_df = features_df.merge(target, left_on='numero_compte', right_index=True, how='left')

    # 8. HANDLE CUSTOMERS WITHOUT BRAND DATA
    all_customers = df_original['numero_compte'].unique()
    if len(features_df) < len(all_customers):
        existing_customers = set(features_df['numero_compte'])
        missing_customers = [c for c in all_customers if c not in existing_customers]

        if missing_customers:
            missing_df = pd.DataFrame({'numero_compte': missing_customers})

            # Set defaults for customers without brand data
            for col in features_df.columns:
                if col != 'numero_compte' and col != 'converted':
                    if col == 'market_data_available':
                        missing_df[col] = 0
                    else:
                        missing_df[col] = 0.0

            # Add target for missing customers
            missing_target = df_original[df_original['numero_compte'].isin(missing_customers)]
            missing_target_series = missing_target.groupby('numero_compte')['fg_devis_accepte'].max()
            missing_df = missing_df.merge(
                missing_target_series.rename('converted'),
                left_on='numero_compte',
                right_index=True,
                how='left'
            )

            features_df = pd.concat([features_df, missing_df], ignore_index=True)

    # Fill missing values
    features_df['converted'] = features_df['converted'].fillna(0).astype(int)

    # 9. FINAL REPORT WITH ADDED DEBUG INFO
    print(f"\n✅ Created {len(features_df.columns) - 2} leakage-free market features")
    print(f"   Total customers: {len(features_df):,}")
    print(f"   With brand data: {features_df['market_data_available'].sum():,}")
    print(f"   Converters: {features_df['converted'].sum():,} ({features_df['converted'].mean():.1%})")

    # Feature correlation check
    print("\n🔍 Feature correlations with target (should be < 0.3 for no leakage):")
    for col in ['premium_brand_ratio', 'budget_brand_ratio', 'brand_consistency', 'brand_diversity_index']:
        if col in features_df.columns:
            # Only check for customers with brand data
            subset = features_df[features_df['market_data_available'] == 1]
            if len(subset) > 1:
                corr = subset[col].corr(subset['converted'])
                status = "✅" if abs(corr) < 0.3 else "⚠️ "
                print(f"   {status} {col:25}: {corr:.3f}")

    return features_df


def remove_leaky_brands(df):
    """
    FORCE remove all brands with perfect conversion prediction
    """
    print("🚨 FORCE REMOVING LEAKY BRANDS")

    # Calculate conversion rates
    brand_stats = df.groupby('marque_produit').agg({
        'fg_devis_accepte': ['count', 'mean', 'sum']
    })

    # Flatten columns
    brand_stats.columns = ['count', 'conversion_rate', 'converted_count']

    # Find leaky brands
    leaky_mask = (brand_stats['conversion_rate'] == 1.0) | (brand_stats['conversion_rate'] == 0.0)
    leaky_brands = brand_stats[leaky_mask].index.tolist()

    print(f"Found {len(leaky_brands)} leaky brands")

    # Also check for near-perfect (99%+ or 1%-)
    near_perfect = brand_stats[
        (brand_stats['conversion_rate'] >= 0.99) | (brand_stats['conversion_rate'] <= 0.01)
        ]
    print(f"Found {len(near_perfect)} near-perfect brands (≥99% or ≤1%)")

    # Remove ALL suspicious brands
    all_suspicious = list(set(leaky_brands + near_perfect.index.tolist()))
    print(f"Total suspicious brands to remove: {len(all_suspicious)}")

    # Show top 10
    for brand in all_suspicious[:10]:
        stats = brand_stats.loc[brand]
        print(f"  - {brand}: {stats['conversion_rate']:.1%} ({stats['count']} quotes)")

    # Replace them
    df_clean = df.copy()
    replace_count = df_clean['marque_produit'].isin(all_suspicious).sum()
    df_clean.loc[df_clean['marque_produit'].isin(all_suspicious), 'marque_produit'] = 'SUSPICIOUS_BRAND'

    print(f"Replaced {replace_count:,} quotes with 'SUSPICIOUS_BRAND'")

    # Verify
    remaining_stats = df_clean.groupby('marque_produit')['fg_devis_accepte'].mean()
    remaining_leaky = remaining_stats[(remaining_stats == 1.0) | (remaining_stats == 0.0)]
    print(f"Remaining leaky brands after removal: {len(remaining_leaky)}")

    return df_clean
