def create_advanced_interaction_features(df):
    """
    Create advanced interaction features based on error analysis insights.
    Designed to be inserted in the feature creation chain.

    Args:
        df: DataFrame with existing features

    Returns:
        DataFrame with new interaction features added
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    df_new = df.copy()

    print("\n" + "=" * 80)
    print("🚀 CREATING ADVANCED INTERACTION FEATURES (Error Analysis Based)")
    print("=" * 80)

    initial_cols = set(df_new.columns)

    # 1. ENGAGEMENT VELOCITY FEATURES
    # Based on interaction analysis suggestion
    if all(col in df_new.columns for col in ['total_quotes', 'avg_days_between_quotes']):
        df_new['engagement_velocity'] = df_new['total_quotes'] / (df_new['avg_days_between_quotes'] + 1)
        df_new['engagement_velocity_scaled'] = pd.qcut(
            df_new['engagement_velocity'].rank(method='first'),
            q=5, labels=['Very Slow', 'Slow', 'Medium', 'Fast', 'Very Fast']
        ).astype(str)

    # 2. PRICE CONSISTENCY FEATURES
    # Based on price_consistency suggestion from interaction analysis
    price_cols = ['min_price', 'max_price', 'avg_price', 'price_range']
    if all(col in df_new.columns for col in ['min_price', 'max_price']):
        # Price stability (inverse of volatility)
        df_new['price_stability'] = 1 - (df_new['price_range'] / (df_new['max_price'] + 1))

        # Price position in range
        if 'avg_price' in df_new.columns:
            df_new['price_position'] = (df_new['avg_price'] - df_new['min_price']) / (df_new['price_range'] + 1)

    # 3. DISCOUNT BEHAVIOR FEATURES (Targeting high-impact error feature)
    if all(col in df_new.columns for col in ['avg_discount_pct', 'price_range', 'max_price']):
        df_new['discount_aggressiveness'] = df_new['avg_discount_pct'] * df_new['price_range']
        df_new['discount_relative_to_price'] = df_new['avg_discount_pct'] * df_new['max_price']

        # Flag aggressive discounters with high quote volume (potential tire-kickers)
        if 'total_quotes' in df_new.columns:
            df_new['aggressive_discount_shopper'] = (
                    (df_new['avg_discount_pct'] > df_new['avg_discount_pct'].median()) &
                    (df_new['price_range'] > df_new['price_range'].median()) &
                    (df_new['total_quotes'] > df_new['total_quotes'].median())
            ).astype(int)

    # 4. DECISION QUALITY FEATURES
    # Based on quote_consistency_score importance
    if all(col in df_new.columns for col in ['quote_consistency_score', 'engagement_density']):
        df_new['decision_quality'] = df_new['quote_consistency_score'] / (df_new['engagement_density'] + 0.01)

        # High quality decisions (consistent + focused)
        df_new['high_quality_decision'] = (
                (df_new['quote_consistency_score'] > df_new['quote_consistency_score'].median()) &
                (df_new['engagement_density'] < df_new['engagement_density'].median())
        ).astype(int)

    # 5. COMPLEXITY PER QUOTE FEATURE
    if all(col in df_new.columns for col in ['solution_complexity_score', 'engagement_density']):
        df_new['complexity_per_quote'] = df_new['solution_complexity_score'] * df_new['engagement_density']

    # 6. FALSE POSITIVE DETECTION FEATURES (Targeting your main issue)

    # Shopping around behavior (tire-kickers)
    if all(col in df_new.columns for col in ['unique_agencies', 'avg_days_between_quotes']):
        df_new['shopping_behavior'] = (
                (df_new['unique_agencies'] > 1) &
                (df_new['avg_days_between_quotes'] < df_new['avg_days_between_quotes'].median())
        ).astype(int)

    # Price sensitivity
    if all(col in df_new.columns for col in ['price_range', 'avg_price']):
        df_new['price_sensitivity'] = (df_new['price_range'] / (df_new['avg_price'] + 1)).clip(0, 10)
        df_new['high_price_sensitivity'] = (df_new['price_sensitivity'] > df_new['price_sensitivity'].median()).astype(
            int)

    # Quick decider but inconsistent
    if all(col in df_new.columns for col in ['is_quick_decider', 'quote_consistency_score']):
        df_new['quick_but_inconsistent'] = (
                (df_new['is_quick_decider'] == 1) &
                (df_new['quote_consistency_score'] < df_new['quote_consistency_score'].median())
        ).astype(int)

    # 7. BRAND LOYALTY INTERACTIONS
    if all(col in df_new.columns for col in ['brand_loyalty_index', 'total_quotes']):
        df_new['loyalty_per_quote'] = df_new['brand_loyalty_index'] / (df_new['total_quotes'] + 1)

        # Highly loyal with few quotes (good signal)
        if 'converted' in df_new.columns:
            # This will be calculated on training only
            pass
        else:
            df_new['loyal_few_quotes'] = (
                    (df_new['brand_loyalty_index'] > 0.9) &
                    (df_new['total_quotes'] <= 2)
            ).astype(int)

    # 8. SEASONAL ENGAGEMENT RATIO (if you have timeline features)
    timeline_cols = ['quotes_per_season_ratio', 'seasonal_std_ratio']
    if any(col in df_new.columns for col in timeline_cols):
        if 'quotes_per_season_ratio' in df_new.columns:
            df_new['seasonal_engagement'] = df_new['quotes_per_season_ratio']

    # 9. CONFIDENCE CALIBRATION FEATURES
    # Features specifically to help with the 40-60% confidence zone

    # Engagement consistency score
    if 'engagement_density' in df_new.columns:
        df_new['engagement_consistency_score'] = 1 - df_new['engagement_density']

    # Decision speed vs complexity
    if all(col in df_new.columns for col in ['is_quick_decider', 'solution_complexity_score']):
        df_new['speed_complexity_ratio'] = df_new['is_quick_decider'] / (df_new['solution_complexity_score'] + 1)

    # 10. BUSINESS VALUE INDICATORS

    # High value potential (based on price)
    if 'max_price' in df_new.columns:
        high_price_threshold = df_new['max_price'].quantile(0.75) if len(df_new) > 100 else df_new['max_price'].median()
        df_new['high_value_potential'] = (df_new['max_price'] > high_price_threshold).astype(int)

    # Quick high-value decider (best case scenario)
    if all(col in df_new.columns for col in ['is_quick_decider', 'high_value_potential']):
        df_new['quick_high_value'] = (
                (df_new['is_quick_decider'] == 1) &
                (df_new['high_value_potential'] == 1)
        ).astype(int)

    # 11. COMBINATION FEATURES FROM ERROR ANALYSIS

    # Top 3 error-driving features interaction
    error_drivers = ['avg_discount_pct', 'max_price', 'quote_consistency_score']
    if all(col in df_new.columns for col in error_drivers):
        # Normalize them first
        scaler = StandardScaler()
        error_features_scaled = scaler.fit_transform(df_new[error_drivers].fillna(0))

        # Create interaction score
        df_new['error_driver_composite'] = (
                error_features_scaled[:, 0] *
                error_features_scaled[:, 1] *
                (1 - error_features_scaled[:, 2])  # Invert consistency (low consistency = more errors)
        )

    # 12. UNCERTAINTY ZONE FEATURES
    # Features specifically designed for the 40-60% prediction confidence range

    # Quote pattern variability
    if 'std_days_between_quotes' in df_new.columns:
        df_new['pattern_variability'] = df_new['std_days_between_quotes'] / (df_new['avg_days_between_quotes'] + 1)

    # Equipment focus
    if all(col in df_new.columns for col in ['equipment_variety_count', 'total_quotes']):
        df_new['equipment_focus_ratio'] = df_new['equipment_variety_count'] / (df_new['total_quotes'] + 1)

    # Print summary
    new_features = set(df_new.columns) - initial_cols
    print(f"\n✅ Created {len(new_features)} new interaction features")
    print("📊 NEW FEATURES ADDED:")
    for i, feat in enumerate(sorted(new_features), 1):
        print(f"  {i:2d}. {feat}")

    # Show sample statistics for key new features
    key_features = [
        'engagement_velocity', 'price_stability', 'discount_aggressiveness',
        'decision_quality', 'shopping_behavior', 'price_sensitivity',
        'error_driver_composite', 'pattern_variability'
    ]

    existing_key_features = [f for f in key_features if f in df_new.columns]
    if existing_key_features:
        print("\n📈 KEY NEW FEATURES STATISTICS:")
        stats = df_new[existing_key_features].describe().T[['mean', 'std', 'min', 'max']]
        print(stats.round(4))

    print("\n" + "=" * 80)

    return df_new