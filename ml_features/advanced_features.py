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


def create_conversion_pattern_features(df):
    """
    Create features targeting actual conversion patterns discovered in error analysis.

    Key insights from data:
    - Converters have HIGH consistency (1.264) but LOWER engagement density (0.829)
    - False positives have PERFECT consistency (1.000) but HIGHER density (0.971)
    - Brand loyalty is paradoxically higher in false positives (0.956 vs 0.904)
    """
    import pandas as pd
    import numpy as np

    df_new = df.copy()

    print("\n" + "=" * 80)
    print("🎯 CREATING CONVERSION PATTERN FEATURES (Based on Error Analysis)")
    print("=" * 80)

    initial_cols = set(df_new.columns)

    # 1. CONSISTENCY OVERLOAD DETECTOR
    # False positives have PERFECT consistency (1.000) - suspicious!
    if 'quote_consistency_score' in df_new.columns:
        # Flag "too perfect" consistency (potential bots or systematic browsers)
        df_new['suspicious_perfect_consistency'] = (
                df_new['quote_consistency_score'] >= 0.99
        ).astype(int)

        # Create consistency tiers
        df_new['consistency_tier'] = pd.cut(
            df_new['quote_consistency_score'],
            bins=[0, 0.8, 0.95, 0.99, 1.0],
            labels=['Low', 'Medium', 'High', 'Perfect']
        ).astype(str)

    # 2. ENGAGEMENT DENSITY SIGNAL
    # Converters have LOWER engagement density - they're more focused
    if 'engagement_density' in df_new.columns:
        # Inverse engagement density (focus score)
        df_new['focus_score'] = 1 / (df_new['engagement_density'] + 0.1)

        # Flag scattered vs focused engagement
        df_new['is_scattered_engagement'] = (
                df_new['engagement_density'] > df_new['engagement_density'].median()
        ).astype(int)

    # 3. BRAND LOYALTY PARADOX
    # Higher loyalty = MORE likely to be false positive?!
    if all(col in df_new.columns for col in ['brand_loyalty_index', 'quote_consistency_score']):
        # Loyal but inconsistent? (True converters pattern)
        df_new['loyal_inconsistent'] = (
                (df_new['brand_loyalty_index'] > 0.9) &
                (df_new['quote_consistency_score'] < 0.99)
        ).astype(int)

        # Loyal but too consistent (False positive pattern)
        df_new['loyal_too_perfect'] = (
                (df_new['brand_loyalty_index'] > 0.9) &
                (df_new['quote_consistency_score'] >= 0.99)
        ).astype(int)

    # 4. COMBINATION SCORES (Your strongest signals combined)
    if all(col in df_new.columns for col in ['quote_consistency_score', 'engagement_density']):
        # TP pattern: High consistency + Low density
        df_new['converter_pattern_score'] = (
                df_new['quote_consistency_score'] / (df_new['engagement_density'] + 0.1)
        )

        # FP pattern: High consistency + High density
        df_new['browser_pattern_score'] = (
                df_new['quote_consistency_score'] * df_new['engagement_density']
        )

    # 5. CONSISTENCY-DENSITY QUADRANTS
    if all(col in df_new.columns for col in ['quote_consistency_score', 'engagement_density']):
        conditions = [
            (df_new['quote_consistency_score'] >= 0.95) & (df_new['engagement_density'] < 0.8),
            (df_new['quote_consistency_score'] >= 0.95) & (df_new['engagement_density'] >= 0.8),
            (df_new['quote_consistency_score'] < 0.95) & (df_new['engagement_density'] < 0.8),
            (df_new['quote_consistency_score'] < 0.95) & (df_new['engagement_density'] >= 0.8)
        ]
        choices = ['High Consistency Focused', 'High Consistency Scattered',
                   'Low Consistency Focused', 'Low Consistency Scattered']
        df_new['consistency_density_profile'] = np.select(conditions, choices, default='Unknown')

    # 6. TEMPORAL PATTERNS (if you have timeline features)
    timeline_features = [col for col in df_new.columns if 'timeline' in col or 'seasonal' in col]
    if timeline_features and 'quote_consistency_score' in df_new.columns:
        # Interaction with timeline features
        for feat in timeline_features[:3]:  # Limit to first 3 to avoid explosion
            df_new[f'consistency_x_{feat}'] = df_new['quote_consistency_score'] * df_new[feat]

    # Print summary
    new_features = set(df_new.columns) - initial_cols
    print(f"\n✅ Created {len(new_features)} conversion pattern features")
    print("\n📊 KEY CONVERSION PATTERNS DISCOVERED:")
    print("  • Perfect consistency (1.000) → SUSPICIOUS (false positive pattern)")
    print("  • Lower engagement density → BETTER (converter pattern)")
    print("  • High brand loyalty + perfect consistency → FALSE POSITIVE risk")
    print("  • High consistency + low density → CONVERTER pattern")

    return df_new


def create_precision_optimization_features(df):
    """
    Create features specifically to reduce false positives while maintaining recall gains.
    """
    import pandas as pd
    import numpy as np

    df_new = df.copy()

    print("\n" + "=" * 80)
    print("🎯 CREATING PRECISION OPTIMIZATION FEATURES")
    print("=" * 80)

    initial_cols = set(df_new.columns)

    # 1. FALSE POSITIVE DETECTOR (based on your FP pattern)
    if all(col in df_new.columns for col in ['quote_consistency_score', 'engagement_density']):
        # FP pattern: High consistency + High density
        df_new['fp_risk_score'] = (
                df_new['quote_consistency_score'] * df_new['engagement_density']
        )

        # Flag high-risk FP candidates
        df_new['high_fp_risk'] = (
                (df_new['quote_consistency_score'] > 0.95) &
                (df_new['engagement_density'] > 0.9)
        ).astype(int)

    # 2. CONSISTENCY THRESHOLD FEATURES
    if 'quote_consistency_score' in df_new.columns:
        # Distance from "perfect" consistency (1.000 is suspicious)
        df_new['consistency_distance_from_perfect'] = 1 - df_new['quote_consistency_score']

        # Flag "too perfect" cases
        df_new['too_perfect_consistency'] = (
                df_new['quote_consistency_score'] > 0.98
        ).astype(int)

    # 3. ENGAGEMENT DENSITY WARNING
    if 'engagement_density' in df_new.columns:
        # High density is risky (FP pattern)
        df_new['high_density_risk'] = (
                df_new['engagement_density'] > df_new['engagement_density'].quantile(0.75)
        ).astype(int)

    # 4. BRAND LOYALTY WARNING (loyalty alone doesn't mean conversion)
    if 'brand_loyalty_index' in df_new.columns:
        df_new['high_loyalty_no_consistency'] = (
                (df_new['brand_loyalty_index'] > 0.9) &
                (df_new['quote_consistency_score'] < 0.8)
        ).astype(int)

    # 5. COMBINATION SCORES (interactions of risk factors)
    risk_factors = ['too_perfect_consistency', 'high_density_risk']
    if all(col in df_new.columns for col in risk_factors):
        df_new['combined_risk_factors'] = (
                df_new['too_perfect_consistency'] + df_new['high_density_risk']
        )

    # 6. CONFIDENCE CALIBRATION
    if 'converter_pattern_score' in df_new.columns and 'fp_risk_score' in df_new.columns:
        # Ratio of converter pattern to FP risk
        df_new['converter_vs_fp_ratio'] = (
                df_new['converter_pattern_score'] / (df_new['fp_risk_score'] + 0.1)
        )

    # 7. THRESHOLD OPTIMIZATION FEATURES
    # Create features that work well at different thresholds
    if 'converter_pattern_score' in df_new.columns:
        df_new['converter_pattern_rank'] = df_new['converter_pattern_score'].rank(pct=True)

    # Print summary
    new_features = set(df_new.columns) - initial_cols
    print(f"\n✅ Created {len(new_features)} precision optimization features")
    print("\n📊 FOCUS: Reducing False Positives")
    print("  • FP Risk Score: Identifies high-consistency + high-density cases")
    print("  • Too Perfect Consistency: Flags suspicious 1.000 scores")
    print("  • Converter vs FP Ratio: Balances both patterns")

    return df_new


def create_price_dominant_features(df):
    """
    Create features centered around price - your strongest predictor.
    """
    import pandas as pd
    import numpy as np

    df_new = df.copy()

    print("\n" + "=" * 80)
    print("💰 CREATING PRICE-DOMINANT FEATURES")
    print("=" * 80)

    initial_cols = set(df_new.columns)

    # 1. PRICE RELATIVITY (how does this customer compare to others?)
    if 'max_price' in df_new.columns:
        # Price percentile within dataset
        df_new['price_percentile'] = df_new['max_price'].rank(pct=True)

        # Price tier
        df_new['price_tier'] = pd.qcut(
            df_new['max_price'],
            q=4,
            labels=['Budget', 'Mid', 'Premium', 'Luxury']
        ).astype(str)

    # 2. PRICE × CONSISTENCY (your top 2 signals combined)
    if all(col in df_new.columns for col in ['max_price', 'quote_consistency_score']):
        df_new['price_x_consistency'] = df_new['max_price'] * df_new['quote_consistency_score']

        # High price + high consistency = strong converter signal
        df_new['premium_consistent'] = (
                (df_new['max_price'] > df_new['max_price'].quantile(0.75)) &
                (df_new['quote_consistency_score'] > 0.9)
        ).astype(int)

    # 3. PRICE × AGENCY (location + price)
    if all(col in df_new.columns for col in ['max_price', 'main_agency']):
        # One-hot encode top agencies
        top_agencies = df_new['main_agency'].value_counts().nlargest(5).index
        for agency in top_agencies:
            df_new[f'price_at_{agency}'] = (
                    df_new['max_price'] * (df_new['main_agency'] == agency).astype(int)
            )

    # 4. PRICE × MODEL COMPLEXITY
    if all(col in df_new.columns for col in ['max_price', 'model_sophistication_score']):
        df_new['price_x_sophistication'] = df_new['max_price'] * df_new['model_sophistication_score']

    # 5. PRICE × DISCOUNT (value perception)
    if all(col in df_new.columns for col in ['max_price', 'avg_discount_pct']):
        df_new['value_score'] = df_new['max_price'] * (1 + df_new['avg_discount_pct'])

    # 6. PRICE RATIOS (compare different price metrics)
    price_pairs = [('max_price', 'min_price'), ('max_price', 'avg_price'), ('avg_price', 'min_price')]
    for p1, p2 in price_pairs:
        if all(col in df_new.columns for col in [p1, p2]):
            df_new[f'{p1}_over_{p2}'] = df_new[p1] / (df_new[p2] + 1)

    # 7. PRICE VOLATILITY (how much they vary their price inquiries)
    if all(col in df_new.columns for col in ['max_price', 'min_price', 'avg_price']):
        df_new['price_spread'] = (df_new['max_price'] - df_new['min_price']) / (df_new['avg_price'] + 1)

    # Print summary
    new_features = set(df_new.columns) - initial_cols
    print(f"\n✅ Created {len(new_features)} price-dominant features")
    print("\n📊 LEVERAGING YOUR STRONGEST SIGNAL:")
    print("  • max_price is your #1 predictor (importance 200)")
    print("  • avg_current_price is #2 (importance 196)")
    print("  • New features combine price with other top signals")

    return df_new