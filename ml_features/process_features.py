import pandas as pd
import numpy as np


def create_process_features(
        df_quotes: pd.DataFrame,
        cutoff_date: str = None,
        customer_col: str = "numero_compte",
        process_col: str = "fg_nouveau_process_relance_devis",
        target_col: str = "fg_devis_accepte"
) -> pd.DataFrame:
    """
    ULTRA-FAST CUSTOMER-LEVEL process features with leakage protection
    VECTORIZED version - 5-10x faster
    """
    import time
    start_time = time.time()

    print("=" * 80)
    print("🚀 ULTRA-FAST CUSTOMER-LEVEL PROCESS FEATURES")
    print("=" * 80)

    # Validate required columns
    required_cols = [customer_col, process_col]
    missing_cols = [col for col in required_cols if col not in df_quotes.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return pd.DataFrame(columns=[customer_col])

    # Make a working copy (minimal columns)
    keep_cols = [customer_col, process_col]
    if cutoff_date and 'dt_creation_devis' in df_quotes.columns:
        keep_cols.append('dt_creation_devis')

    df = df_quotes[keep_cols].copy()

    # --------------------------------------------------------------------
    # 1. APPLY TEMPORAL FILTER (VECTORIZED)
    # --------------------------------------------------------------------
    if cutoff_date and 'dt_creation_devis' in df.columns:
        cutoff_date = pd.to_datetime(cutoff_date)
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Vectorized filtering
        mask = df['dt_creation_devis'] <= cutoff_date
        df = df[mask].copy()

        print(f"📅 Applied temporal cutoff: {cutoff_date.date()}")
        print(f"📊 Quotes after filtering: {len(df):,}")

    if len(df) == 0:
        print("⚠️ No data after temporal filtering")
        return pd.DataFrame(columns=[customer_col])

    print(f"Processing {len(df):,} quotes for {df[customer_col].nunique():,} customers")

    # --------------------------------------------------------------------
    # 2. SINGLE-PASS VECTORIZED GROUPBY
    # --------------------------------------------------------------------
    print("👥 Vectorized aggregation...")

    # Convert process_col to numeric once
    df['process_flag'] = pd.to_numeric(df[process_col], errors='coerce')

    # Base aggregation dictionary
    agg_dict = {
        'total_quotes': ('process_flag', 'size'),  # Faster than count
        'new_process_count': ('process_flag', lambda x: (x == 1).sum()),
        'old_process_count': ('process_flag', lambda x: (x == 0).sum()),
    }

    # Add temporal aggregations if dates available
    if 'dt_creation_devis' in df.columns:
        df['quote_date'] = df['dt_creation_devis']
        agg_dict.update({
            'first_quote_date': ('quote_date', 'min'),
            'last_quote_date': ('quote_date', 'max'),
        })

    # Perform single groupby with all aggregations
    grouped = df.groupby(customer_col).agg(**agg_dict).reset_index()

    # --------------------------------------------------------------------
    # 3. VECTORIZED FEATURE CALCULATION
    # --------------------------------------------------------------------
    print("⚡ Vectorized feature calculation...")

    # Process adoption rate
    grouped['process_adoption_rate'] = grouped['new_process_count'] / grouped['total_quotes']

    # Has ever used process
    grouped['has_ever_used_process'] = (grouped['new_process_count'] > 0).astype(int)

    # Process consistency (always same or never same)
    always_new = (grouped['new_process_count'] == grouped['total_quotes']).astype(float)
    always_old = (grouped['old_process_count'] == grouped['total_quotes']).astype(float)
    grouped['process_consistency'] = always_new + always_old

    # Process volatility (using variance formula)
    p = grouped['process_adoption_rate']
    n = grouped['total_quotes']
    # Variance of Bernoulli distribution = p*(1-p)
    # But we want volatility between 0-1
    grouped['process_volatility'] = p * (1 - p)

    # Process preference categories using np.select (vectorized)
    conditions = [
        grouped['process_adoption_rate'] == 0,
        grouped['process_adoption_rate'] == 1,
        (grouped['process_adoption_rate'] > 0) & (grouped['process_adoption_rate'] < 1)
    ]
    choices = [0, 2, 1]
    grouped['process_preference'] = np.select(conditions, choices, default=1)

    # --------------------------------------------------------------------
    # 4. CURRENT PROCESS USAGE (requires date sorting)
    # --------------------------------------------------------------------
    if 'dt_creation_devis' in df.columns:
        print("  → Getting latest process usage...")

        # Get the most recent process flag per customer (vectorized)
        df_sorted = df.sort_values([customer_col, 'dt_creation_devis'])
        latest_idx = df_sorted.groupby(customer_col)['dt_creation_devis'].idxmax()
        latest_process = df_sorted.loc[latest_idx, [customer_col, 'process_flag']]
        latest_process = latest_process.rename(columns={'process_flag': 'current_process_usage'})

        # Merge with grouped data
        grouped = pd.merge(grouped, latest_process, on=customer_col, how='left')
        grouped['current_process_usage'] = (grouped['current_process_usage'] == 1).astype(int)
    else:
        grouped['current_process_usage'] = ((grouped['new_process_count'] / grouped['total_quotes']) > 0.5).astype(int)

    # --------------------------------------------------------------------
    # 5. RECENT PROCESS ADOPTION (last 3 quotes)
    # --------------------------------------------------------------------
    if 'dt_creation_devis' in df.columns and len(df) > 0:
        print("  → Calculating recent adoption...")

        # Sort and get last 3 quotes per customer
        df_sorted = df.sort_values([customer_col, 'dt_creation_devis'])
        last_3_idx = df_sorted.groupby(customer_col).tail(3).index
        last_3_df = df_sorted.loc[last_3_idx]

        # Aggregate last 3 quotes
        recent_stats = last_3_df.groupby(customer_col)['process_flag'].agg(
            recent_new_count=lambda x: (x == 1).sum(),
            recent_total=('process_flag', 'size')
        ).reset_index()

        # Calculate recent adoption rate
        recent_stats['recent_process_adoption'] = recent_stats['recent_new_count'] / recent_stats['recent_total']
        recent_stats['recent_process_adoption'] = recent_stats['recent_process_adoption'].fillna(0)

        # Merge with grouped data
        grouped = pd.merge(grouped, recent_stats[[customer_col, 'recent_process_adoption']],
                           on=customer_col, how='left')
        grouped['recent_process_adoption'] = grouped['recent_process_adoption'].fillna(grouped['process_adoption_rate'])
    else:
        grouped['recent_process_adoption'] = grouped['process_adoption_rate']

    # --------------------------------------------------------------------
    # 6. TEMPORAL FEATURES (if dates available)
    # --------------------------------------------------------------------
    if 'first_quote_date' in grouped.columns and cutoff_date:
        print("  → Calculating temporal features...")

        grouped['days_since_first_quote'] = (cutoff_date - grouped['first_quote_date']).dt.days
        grouped['days_since_last_quote'] = (cutoff_date - grouped['last_quote_date']).dt.days

        # Process adoption trend (if enough quotes)
        if 'dt_creation_devis' in df.columns:
            # Split each customer's quotes into early vs late
            df['quote_rank'] = df.groupby(customer_col)['dt_creation_devis'].rank(method='first')
            df['is_early'] = df['quote_rank'] <= df.groupby(customer_col)['quote_rank'].transform('max') / 2

            # Calculate early vs late adoption
            early_adoption = df[df['is_early']].groupby(customer_col)['process_flag'].apply(
                lambda x: (x == 1).mean() if len(x) > 0 else 0
            ).reset_index(name='early_adoption_rate')

            late_adoption = df[~df['is_early']].groupby(customer_col)['process_flag'].apply(
                lambda x: (x == 1).mean() if len(x) > 0 else 0
            ).reset_index(name='late_adoption_rate')

            # Merge and calculate trend
            trend_df = pd.merge(early_adoption, late_adoption, on=customer_col, how='outer').fillna(0)
            trend_df['process_adoption_trend'] = trend_df['late_adoption_rate'] - trend_df['early_adoption_rate']

            grouped = pd.merge(grouped, trend_df[[customer_col, 'process_adoption_trend']],
                               on=customer_col, how='left')
            grouped['process_adoption_trend'] = grouped['process_adoption_trend'].fillna(0)

    # --------------------------------------------------------------------
    # 7. DERIVED FEATURES (VECTORIZED)
    # --------------------------------------------------------------------
    print("  → Creating derived features...")

    # Process engagement intensity
    if 'days_since_first_quote' in grouped.columns:
        mask = grouped['days_since_first_quote'] > 0
        grouped.loc[mask, 'process_engagement_intensity'] = (
                grouped.loc[mask, 'total_quotes'] / grouped.loc[mask, 'days_since_first_quote']
        )
        grouped['process_engagement_intensity'] = grouped['process_engagement_intensity'].fillna(0)

    # Process switch indicator (vectorized)
    grouped['recent_process_switch'] = (
            (grouped['recent_process_adoption'] > 0.8) &
            (grouped['process_adoption_rate'] < 0.5)
    ).astype(int)

    # Process confidence score
    grouped['process_confidence_score'] = (
            grouped['process_consistency'] * 0.6 +
            np.clip(grouped['process_adoption_rate'], 0, 1) * 0.4
    ).clip(0, 1)

    # Process stability index (composite)
    grouped['process_stability_index'] = (
            (1 - grouped['process_volatility']) * 0.4 +
            grouped['process_consistency'] * 0.3 +
            (grouped['total_quotes'] >= 3).astype(float) * 0.3
    ).clip(0, 1)

    # --------------------------------------------------------------------
    # 8. CLEANUP AND FINALIZATION
    # --------------------------------------------------------------------
    print("📝 Finalizing features...")

    # Drop intermediate columns
    columns_to_drop = ['new_process_count', 'old_process_count']
    if 'first_quote_date' in grouped.columns:
        columns_to_drop.extend(['first_quote_date', 'last_quote_date'])

    grouped = grouped.drop(columns=columns_to_drop, errors='ignore')

    # Fill NaN values
    numeric_cols = grouped.select_dtypes(include=[np.number]).columns
    grouped[numeric_cols] = grouped[numeric_cols].fillna(0)

    # Ensure process_adoption_rate is between 0-1
    grouped['process_adoption_rate'] = grouped['process_adoption_rate'].clip(0, 1)
    grouped['recent_process_adoption'] = grouped['recent_process_adoption'].clip(0, 1)

    # --------------------------------------------------------------------
    # 9. FINAL REPORT
    # --------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    feature_count = len([col for col in grouped.columns if col != customer_col])

    print(f"\n✅ Created {feature_count} process features for {len(grouped):,} customers")
    print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")

    print("\n📊 FEATURE SUMMARY:")
    print("-" * 50)

    summary_stats = [
        ('process_adoption_rate', 'Adoption Rate'),
        ('has_ever_used_process', 'Ever Used (%)'),
        ('process_consistency', 'Consistency'),
        ('process_confidence_score', 'Confidence Score'),
    ]

    for col, name in summary_stats:
        if col in grouped.columns:
            if col == 'has_ever_used_process':
                positive_pct = grouped[col].mean() * 100
                print(f"{name:25} : {positive_pct:.1f}%")
            else:
                mean_val = grouped[col].mean()
                std_val = grouped[col].std()
                print(f"{name:25} : mean = {mean_val:.3f}, std = {std_val:.3f}")

    # Rename customer column back to original name
    grouped = grouped.rename(columns={customer_col: 'numero_compte'})

    return grouped