import numpy as np
import pandas as pd


def create_timeline_features(
        df: pd.DataFrame,
        customer_col: str = "numero_compte",
        safe_date_col: str = "dt_creation_devis",  # Primary safe date
        first_quote_col: str = None,  # Optional: dt_prem_devis (use with caution!)
) -> pd.DataFrame:
    """
    LEAKAGE-SAFE timeline features using only safe date columns
    """
    print("=" * 80)
    print("CREATING SAFE TIMELINE FEATURES (NO LEAKAGE)")
    print("=" * 80)

    # Check essential column
    if safe_date_col not in df.columns:
        print(f"⚠️ WARNING: '{safe_date_col}' column not found")
        return pd.DataFrame(columns=[customer_col])

    print(f"Using safe date column: '{safe_date_col}'")
    print(f"Processing {df[customer_col].nunique():,} customers")

    # Make working copy with ONLY safe columns
    # ALWAYS sort chronologically first
    df_work = df[[customer_col, safe_date_col]].copy()
    df_work[safe_date_col] = pd.to_datetime(df_work[safe_date_col], errors='coerce')

    # FORCE chronological order - this is CRITICAL
    df_work = df_work.sort_values([customer_col, safe_date_col]).reset_index(drop=True)

    # Add first quote date if explicitly requested AND safe
    if first_quote_col and first_quote_col in df.columns:
        print(f"⚠️ Using '{first_quote_col}' - verify this is known at quote time")
        df_work[first_quote_col] = df[first_quote_col]

    # Convert dates
    df_work[safe_date_col] = pd.to_datetime(df_work[safe_date_col], errors='coerce')
    if first_quote_col and first_quote_col in df_work.columns:
        df_work[first_quote_col] = pd.to_datetime(df_work[first_quote_col], errors='coerce')

    # Remove invalid dates
    valid_mask = df_work[safe_date_col].notna()
    df_work = df_work[valid_mask].copy()

    if len(df_work) == 0:
        print("⚠️ No valid date data available")
        return pd.DataFrame(columns=[customer_col])

    # Sort chronologically
    df_work = df_work.sort_values([customer_col, safe_date_col]).reset_index(drop=True)

    # ========== SAFE FEATURES ONLY ==========

    # Basic sequence
    df_work['quote_seq'] = df_work.groupby(customer_col).cumcount()

    # Time between quotes (safe)
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

    result = result.join(time_stats)

    # Engagement consistency
    result['engagement_consistency'] = 1 - np.minimum(
        result['time_between_quotes_std'] / result['avg_days_between_quotes'].replace(0, 1),
        1
    )

    # Seasonality (safe - only uses month)
    df_work['month'] = df_work[safe_date_col].dt.month
    month_counts = df_work.groupby([customer_col, 'month']).size().unstack(fill_value=0)

    result['peak_engagement_month'] = month_counts.idxmax(axis=1)
    result['month_concentration'] = month_counts.max(axis=1) / month_counts.sum(axis=1)

    print(f"\n✅ Created {len(result.columns)} SAFE timeline features")
    print("   REMOVED: company_tenure_days, first_quote_recency (potential leakage)")

    return result.reset_index()


def create_advanced_timeline_features(df):
    """
    Create more sophisticated timeline features:
    1. Quote acceleration/deceleration patterns
    2. Day-of-week and time-of-day patterns
    3. Holiday/event proximity
    4. Quote clustering analysis
    """
    print("Creating ADVANCED timeline features...")

    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()
    df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

    advanced_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        quote_dates = customer_data['dt_creation_devis'].dropna()

        if len(quote_dates) < 2:
            # Not enough data for advanced patterns
            features.update({
                'quote_acceleration': 0,
                'day_of_week_concentration': 0,
                'has_quote_clusters': 0,
                'time_based_engagement_score': 0
            })
        else:
            # ========== FEATURE 1: QUOTE ACCELERATION ==========
            time_diffs = quote_dates.diff().dropna().dt.days.values

            # Calculate if time between quotes is decreasing (acceleration)
            if len(time_diffs) > 1:
                # Linear trend of time differences
                x = np.arange(len(time_diffs))
                slope, _ = np.polyfit(x, time_diffs, 1)
                features['quote_acceleration'] = -slope  # Negative slope = acceleration

                # Acceleration indicator
                features['is_accelerating'] = 1 if slope < -0.1 else 0
                features['is_decelerating'] = 1 if slope > 0.1 else 0
            else:
                features['quote_acceleration'] = 0
                features['is_accelerating'] = 0
                features['is_decelerating'] = 0

            # ========== FEATURE 2: DAY-OF-WEEK PATTERNS ==========
            weekdays = quote_dates.dt.dayofweek  # Monday=0, Sunday=6

            weekday_counts = weekdays.value_counts()
            if len(weekday_counts) > 0:
                # Most common weekday
                features['peak_weekday'] = weekday_counts.index[0]

                # Concentration on specific weekdays
                features['weekday_concentration'] = weekday_counts.iloc[0] / len(weekdays)

                # Business vs weekend
                business_days = sum(1 for d in weekdays if d < 5)  # Monday-Friday
                features['business_day_ratio'] = business_days / len(weekdays)
            else:
                features['peak_weekday'] = 0
                features['weekday_concentration'] = 0
                features['business_day_ratio'] = 0

            # ========== FEATURE 3: QUOTE CLUSTERING ==========
            # Detect if quotes come in clusters (multiple quotes close together)
            cluster_threshold = 3  # days
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

            # ========== FEATURE 4: TIME-BASED ENGAGEMENT SCORE ==========
            engagement_components = []

            # Component 1: Acceleration (accelerating is good)
            if features.get('is_accelerating', 0) == 1:
                engagement_components.append(0.8)
            elif features.get('is_decelerating', 0) == 1:
                engagement_components.append(0.2)
            else:
                engagement_components.append(0.5)

            # Component 2: Business day focus (business days might indicate seriousness)
            if features.get('business_day_ratio', 0) > 0.8:
                engagement_components.append(0.7)  # Strong business focus
            elif features.get('business_day_ratio', 0) < 0.2:
                engagement_components.append(0.3)  # Mostly weekend
            else:
                engagement_components.append(0.5)  # Mixed

            # Component 3: Quote clustering (clusters might indicate project focus)
            if features.get('has_quote_clusters', 0) == 1:
                engagement_components.append(0.6)  # Shows focused interest
            else:
                engagement_components.append(0.4)  # More sporadic

            features['time_based_engagement_score'] = np.mean(engagement_components) if engagement_components else 0.5

        advanced_features.append(features)

    return pd.DataFrame(advanced_features)


def create_timeline_interaction_features(timeline_df, brand_df, equipment_df):
    """
    Create interaction features between timeline and other pillars
    """
    # Merge dataframes
    merged = pd.merge(timeline_df, brand_df, on='numero_compte', how='left')
    merged = pd.merge(merged, equipment_df, on='numero_compte', how='left')

    interaction_features = []

    for idx, row in merged.iterrows():
        features = {'numero_compte': row['numero_compte']}

        # Interaction 1: Brand loyalty + Temporal consistency
        if 'brand_loyalty_index' in row and 'temporal_consistency_score' in row:
            features['loyal_consistent_customer'] = row['brand_loyalty_index'] * row['temporal_consistency_score']

        # Interaction 2: Equipment maturity + Seasonality
        if 'equipment_maturity_level' in row and 'seasonal_concentration' in row:
            features['mature_seasonal_focus'] = row['equipment_maturity_level'] * row['seasonal_concentration']

        interaction_features.append(features)

    return pd.DataFrame(interaction_features)