import numpy as np
import pandas as pd

from ml_features.globals import PREMIUM_BRANDS, BUDGET_BRANDS


def create_brand_features(df):
    print("=" * 80)
    print("CREATING BRAND FEATURES")
    print("=" * 80)

    if 'marque_produit' not in df.columns:
        print("⚠️ No brand data")
        return pd.DataFrame(columns=['numero_compte'])

    # Sort once
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)

    print(f"Processing {len(df):,} quotes for {df['numero_compte'].nunique():,} customers")

    # SINGLE GROUPBY to get all sequences
    print("👥 Single groupby aggregation...")

    customer_groups = df.groupby('numero_compte')['marque_produit'].apply(list)
    customer_ids = customer_groups.index.values
    brand_sequences = customer_groups.values
    n_customers = len(customer_ids)

    print(f"  Processing {n_customers:,} customers with brand data")

    # VECTORIZED FEATURE CALCULATION
    print("⚡ Vectorized feature calculation...")

    # Initialize arrays
    brand_data_available = np.ones(n_customers, dtype=int)
    brand_loyalty_index = np.zeros(n_customers, dtype=float)
    brand_switches = np.zeros(n_customers, dtype=int)
    prefers_premium_brand = np.zeros(n_customers, dtype=int)
    prefers_budget_brand = np.zeros(n_customers, dtype=int)
    brand_consistency = np.zeros(n_customers, dtype=int)
    brand_persistence_ratio = np.ones(n_customers, dtype=float)
    brand_convergence = np.ones(n_customers, dtype=int)

    # Process in batches for memory efficiency
    batch_size = 1000

    for i in range(0, n_customers, batch_size):
        batch_end = min(i + batch_size, n_customers)

        for j in range(i, batch_end):
            seq = brand_sequences[j]
            seq_len = len(seq)

            if seq_len == 0:
                brand_data_available[j] = 0
                continue

            # Convert to numpy array for fast operations
            seq_array = np.array(seq)

            # FEATURE 1: Brand loyalty index (most common brand ratio)
            unique, counts = np.unique(seq_array, return_counts=True)
            brand_loyalty_index[j] = counts.max() / seq_len

            # FEATURE 2: Brand switches (unique brands - 1)
            brand_switches[j] = max(0, len(unique) - 1)

            # FEATURE 3: Premium/Budget preference (most common brand)
            top_brand = unique[counts.argmax()]
            prefers_premium_brand[j] = 1 if top_brand in PREMIUM_BRANDS else 0
            prefers_budget_brand[j] = 1 if top_brand in BUDGET_BRANDS else 0

            # FEATURE 4: Brand consistency (all same brand)
            brand_consistency[j] = 1 if len(unique) == 1 else 0

            # FEATURE 5: Persistence ratio (for multi-quote)
            if seq_len > 1:
                changes = np.sum(seq_array[1:] != seq_array[:-1])
                brand_persistence_ratio[j] = 1 - (changes / (seq_len - 1))

            # FEATURE 6: Brand convergence
            if seq_len > 1 and len(unique) > 1:
                # Check if converges to single brand
                mid_point = seq_len // 2
                first_half = seq_array[:mid_point]
                second_half = seq_array[mid_point:]

                first_unique = np.unique(first_half)
                second_unique = np.unique(second_half)

                # Converges if: starts with multiple, ends with single
                starts_multiple = len(first_unique) > 1
                ends_single = len(second_unique) == 1
                brand_convergence[j] = 1 if (starts_multiple and ends_single) else 0

    print("✅ Vectorized calculations complete")

    # CREATE FINAL DATAFRAME
    print("📝 Creating final DataFrame...")

    result = pd.DataFrame({
        'numero_compte': customer_ids,
        'brand_data_available': brand_data_available,
        'brand_loyalty_index': brand_loyalty_index,
        'brand_switches': brand_switches,
        'prefers_premium_brand': prefers_premium_brand,
        'prefers_budget_brand': prefers_budget_brand,
        'brand_consistency': brand_consistency,
        'brand_persistence_ratio': brand_persistence_ratio,
        'brand_convergence': brand_convergence
    })

    # ADD CUSTOMERS WITHOUT BRAND DATA
    all_customers = df['numero_compte'].unique()
    if len(result) < len(all_customers):
        existing_customers = set(customer_ids)
        missing_customers = [c for c in all_customers if c not in existing_customers]

        if missing_customers:
            missing_df = pd.DataFrame({'numero_compte': missing_customers})

            # Default values for customers without brand data
            default_values = {
                'brand_data_available': 0,
                'brand_loyalty_index': 0,
                'brand_switches': 0,
                'prefers_premium_brand': 0,
                'prefers_budget_brand': 0,
                'brand_consistency': 0,
                'brand_persistence_ratio': 1,
                'brand_convergence': 1
            }

            for col, val in default_values.items():
                missing_df[col] = val

            result = pd.concat([result, missing_df], ignore_index=True)

    # FINAL REPORT
    print(f"\n✅ Created {len(result.columns) - 1} brand features")
    print(f"   Total customers: {len(result):,}")
    print(f"   With brand data: {result['brand_data_available'].sum():,}")

    # Quick summary
    print("\n📊 FEATURE SUMMARY:")
    print("-" * 50)
    for col in ['brand_loyalty_index', 'brand_switches', 'brand_consistency',
                'prefers_premium_brand', 'prefers_budget_brand']:
        if col in result.columns:
            mean_val = result[col].mean()
            print(f"{col:25} : mean = {mean_val:.3f}")

    return result