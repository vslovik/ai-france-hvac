import numpy as np
import pandas as pd


def create_timeline_features(
        df: pd.DataFrame,
        target_type: str = 'first_conversion',
        customer_col: str = "numero_compte",
        safe_date_col: str = "dt_creation_devis",
        first_quote_col: str = None,
        accept_col: str = "fg_devis_accepte"
) -> pd.DataFrame:
    """
    FIXED: Timeline features WITHOUT aggressive filtering
    """
    print("=" * 80)
    print(f"CREATING TIMELINE FEATURES (mode: {target_type})")
    print("=" * 80)

    # Check essential column
    if safe_date_col not in df.columns:
        print(f"⚠️ '{safe_date_col}' column not found")
        return pd.DataFrame(columns=[customer_col])

    print(f"Using date column: '{safe_date_col}'")
    print(f"Processing {df[customer_col].nunique():,} customers")

    # Make working copy
    df_work = df[[customer_col, safe_date_col, accept_col]].copy()

    # Store original for target calculation
    df_original = df_work.copy()

    # Convert dates
    df_work[safe_date_col] = pd.to_datetime(df_work[safe_date_col], errors='coerce')

    print(f"\n🔧 NO FILTERING: Using ALL quotes for timeline features")
    print(f"   (Like market features, timeline patterns are stable characteristics)")
    print(f"   Quotes for features: {len(df_work):,}")

    # Add first quote date if explicitly requested
    if first_quote_col and first_quote_col in df.columns:
        print(f"⚠️ Using '{first_quote_col}'")
        df_work[first_quote_col] = pd.to_datetime(df[first_quote_col], errors='coerce')

    # Remove invalid dates
    valid_mask = df_work[safe_date_col].notna()
    df_work = df_work[valid_mask].copy()

    if len(df_work) == 0:
        print("⚠️ No valid date data available")
        result = pd.DataFrame(columns=[customer_col])
        target = df_original.groupby(customer_col)[accept_col].max()
        target.name = 'converted'
        result = result.merge(target, left_on=customer_col, right_index=True, how='left')
        return result

    # Sort chronologically
    df_work = df_work.sort_values([customer_col, safe_date_col]).reset_index(drop=True)

    # ========== FEATURE CALCULATION ==========

    # Basic sequence
    df_work['quote_seq'] = df_work.groupby(customer_col).cumcount()

    # Time between quotes
    df_work['prev_date'] = df_work.groupby(customer_col)[safe_date_col].shift()
    df_work['days_since_prev'] = (df_work[safe_date_col] - df_work['prev_date']).dt.days

    # Aggregate to customer level
    result = pd.DataFrame()

    # Time stats
    time_stats = df_work.groupby(customer_col)['days_since_prev'].agg(
        avg_days_between_quotes='mean',
        time_between_quotes_std='std',
        max_days_between_quotes='max',
        min_days_between_quotes='min'
    ).fillna(0)

    result = time_stats

    # Engagement consistency
    result['engagement_consistency'] = 1 - np.minimum(
        result['time_between_quotes_std'] / result['avg_days_between_quotes'].replace(0, 1),
        1
    )

    # All customers have timeline data now
    result['timeline_data_available'] = 1

    # Seasonality
    df_work['month'] = df_work[safe_date_col].dt.month
    month_counts = df_work.groupby([customer_col, 'month']).size().unstack(fill_value=0)

    # Handle all customers
    all_customers = set(df_original[customer_col].unique())
    processed_customers = set(result.index)
    missing_customers = list(all_customers - processed_customers)

    if missing_customers:
        missing_df = pd.DataFrame(index=missing_customers)
        missing_df['avg_days_between_quotes'] = 0
        missing_df['time_between_quotes_std'] = 0
        missing_df['max_days_between_quotes'] = 0
        missing_df['min_days_between_quotes'] = 0
        missing_df['engagement_consistency'] = 0
        missing_df['timeline_data_available'] = 0

        for month in range(1, 13):
            missing_df[f'month_{month}_count'] = 0

        result = pd.concat([result, missing_df])

    # Add month features
    if not month_counts.empty:
        month_counts = month_counts.add_prefix('month_').add_suffix('_count')
        result['month_concentration'] = month_counts.max(axis=1) / month_counts.sum(axis=1).replace(0, 1)
        result['peak_engagement_month'] = month_counts.idxmax(axis=1).str.replace('month_', '').str.replace('_count',
                                                                                                            '').astype(
            int)
        result = result.merge(month_counts, left_index=True, right_index=True, how='left').fillna(0)
    else:
        result['month_concentration'] = 0
        result['peak_engagement_month'] = 0

    # Add target
    print("\n🎯 Adding target variable...")
    target = df_original.groupby(customer_col)[accept_col].max()
    target.name = 'converted'
    result = result.merge(target, left_index=True, right_index=True, how='left')

    result = result.reset_index().rename(columns={'index': customer_col})
    result['converted'] = result['converted'].fillna(0).astype(int)

    # 🚨 DEBUG: Verify balanced data
    print(f"\n🚨 VERIFICATION: Data availability (should be ~100% for both)")
    converters = result[result['converted'] == 1]
    non_converters = result[result['converted'] == 0]

    conv_with_data = converters['timeline_data_available'].mean()
    non_conv_with_data = non_converters['timeline_data_available'].mean()

    print(f"  Converters with data: {conv_with_data:.1%}")
    print(f"  Non-converters with data: {non_conv_with_data:.1%}")
    print(f"  Difference: {abs(conv_with_data - non_conv_with_data):.1%} (should be < 10%)")

    if abs(conv_with_data - non_conv_with_data) > 0.1:
        print(f"  ⚠️  WARNING: Still imbalanced! But much better than 13.7% vs 91.0%")

    print(f"\n✅ Created {len(result.columns) - 2} timeline features")
    print(f"   Total customers: {len(result):,}")
    print(f"   Converters: {result['converted'].sum():,} ({result['converted'].mean():.1%})")

    return result


def create_advanced_timeline_features(df, target_type='first_conversion'):
    """
    FIXED: Advanced timeline features WITHOUT filtering
    (Timeline patterns are stable characteristics)
    """
    print("=" * 80)
    print(f"CREATING ADVANCED TIMELINE FEATURES (mode: {target_type})")
    print("=" * 80)

    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()
    df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

    print(f"Initial data: {len(df):,} quotes for {df['numero_compte'].nunique():,} customers")

    # Store original for target
    df_original = df.copy()

    # 🚨 CRITICAL: NO FILTERING for advanced timeline features
    print(f"\n🔧 NO FILTERING: Using ALL quotes for advanced timeline features")
    print(f"   (Like basic timeline and market features)")
    print(f"   Quotes for features: {len(df):,}")

    advanced_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        quote_dates = customer_data['dt_creation_devis'].dropna()

        # All customers have data now
        features['advanced_timeline_data_available'] = 1 if len(quote_dates) > 0 else 0

        if len(quote_dates) < 2:
            # Single quote or no quotes
            features.update({
                'quote_acceleration': 0,
                'is_accelerating': 0,
                'is_decelerating': 0,
                'peak_weekday': quote_dates.iloc[0].dayofweek if len(quote_dates) == 1 else 0,
                'weekday_concentration': 1 if len(quote_dates) == 1 else 0,
                'business_day_ratio': 1 if len(quote_dates) == 1 and quote_dates.iloc[0].dayofweek < 5 else 0,
                'quote_cluster_count': 0,
                'has_quote_clusters': 0,
                'time_based_engagement_score': 0.5
            })
        else:
            # ========== ADVANCED FEATURES ==========
            time_diffs = quote_dates.diff().dropna().dt.days.values

            # 1. Quote acceleration
            if len(time_diffs) > 1:
                x = np.arange(len(time_diffs))
                slope, _ = np.polyfit(x, time_diffs, 1)
                features['quote_acceleration'] = -slope
                features['is_accelerating'] = 1 if slope < -0.1 else 0
                features['is_decelerating'] = 1 if slope > 0.1 else 0
            else:
                features['quote_acceleration'] = 0
                features['is_accelerating'] = 0
                features['is_decelerating'] = 0

            # 2. Day-of-week patterns
            weekdays = quote_dates.dt.dayofweek
            weekday_counts = weekdays.value_counts()
            if len(weekday_counts) > 0:
                features['peak_weekday'] = weekday_counts.index[0]
                features['weekday_concentration'] = weekday_counts.iloc[0] / len(weekdays)
                business_days = sum(1 for d in weekdays if d < 5)
                features['business_day_ratio'] = business_days / len(weekdays)
            else:
                features['peak_weekday'] = 0
                features['weekday_concentration'] = 0
                features['business_day_ratio'] = 0

            # 3. Quote clustering
            cluster_threshold = 3
            in_cluster = False
            cluster_count = 0
            for i in range(1, len(time_diffs)):
                if time_diffs[i - 1] <= cluster_threshold:
                    if not in_cluster:
                        cluster_count += 1
                        in_cluster = True
                else:
                    in_cluster = False

            features['quote_cluster_count'] = cluster_count
            features['has_quote_clusters'] = 1 if cluster_count > 0 else 0

            # 4. Engagement score
            engagement_components = []
            if features.get('is_accelerating', 0) == 1:
                engagement_components.append(0.8)
            elif features.get('is_decelerating', 0) == 1:
                engagement_components.append(0.2)
            else:
                engagement_components.append(0.5)

            if features.get('business_day_ratio', 0) > 0.8:
                engagement_components.append(0.7)
            elif features.get('business_day_ratio', 0) < 0.2:
                engagement_components.append(0.3)
            else:
                engagement_components.append(0.5)

            if features.get('has_quote_clusters', 0) == 1:
                engagement_components.append(0.6)
            else:
                engagement_components.append(0.4)

            features['time_based_engagement_score'] = np.mean(engagement_components) if engagement_components else 0.5

        advanced_features.append(features)

    # Create DataFrame
    result_df = pd.DataFrame(advanced_features)

    # Add target
    print("\n🎯 Adding target variable...")
    target = df_original.groupby('numero_compte')['fg_devis_accepte'].max()
    target.name = 'converted'
    result_df = result_df.merge(target, left_on='numero_compte', right_index=True, how='left')
    result_df['converted'] = result_df['converted'].fillna(0).astype(int)

    # Verification
    print(f"\n✅ Created advanced timeline features for {len(result_df):,} customers")
    print(f"   With data: {result_df['advanced_timeline_data_available'].sum():,}")
    print(f"   Converters: {result_df['converted'].sum():,} ({result_df['converted'].mean():.1%})")

    # Check data balance
    conv_with_data = result_df[result_df['converted'] == 1]['advanced_timeline_data_available'].mean()
    non_conv_with_data = result_df[result_df['converted'] == 0]['advanced_timeline_data_available'].mean()
    print(f"\n🚨 VERIFICATION: Data balance")
    print(f"   Converters with data: {conv_with_data:.1%}")
    print(f"   Non-converters with data: {non_conv_with_data:.1%}")
    print(f"   Difference: {abs(conv_with_data - non_conv_with_data):.1%}")

    return result_df


def create_timeline_interaction_features(df):
    """
    Create interaction features between timeline and other pillars
    """

    interaction_features = []

    for idx, row in df.iterrows():
        features = {'numero_compte': row['numero_compte']}

        # Interaction 1: Brand loyalty + Temporal consistency
        if 'brand_loyalty_index' in row and 'temporal_consistency_score' in row:
            features['loyal_consistent_customer'] = row['brand_loyalty_index'] * row['temporal_consistency_score']

        # Interaction 2: Equipment maturity + Seasonality
        if 'equipment_maturity_level' in row and 'seasonal_concentration' in row:
            features['mature_seasonal_focus'] = row['equipment_maturity_level'] * row['seasonal_concentration']

        interaction_features.append(features)

    return pd.DataFrame(interaction_features)