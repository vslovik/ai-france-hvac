def create_engagement_interaction_features(df, y_target=None):
    """
    Create efficiency-focused interaction features targeting the core problem:
    Models penalize short consideration cycles and simple solutions

    Args:
        df: DataFrame with features
        y_target: Optional target series for correlation calculation
    """
    print("\n" + "=" * 60)
    print("🎯 EFFICIENCY-FOCUSED FEATURE ENGINEERING")
    print("=" * 60)

    efficiency_features = {}

    # === YOUR SUGGESTED FEATURES ===
    print("\n📈 ADDING YOUR SUGGESTED FEATURES:")

    # 1. Engagement pattern features
    if 'total_quotes' in df.columns and 'avg_days_between_quotes' in df.columns:
        df['engagement_speed_score'] = df['total_quotes'] / (df['avg_days_between_quotes'] + 1)
        efficiency_features['engagement_speed_score'] = 'total_quotes / (avg_days + 1)'
        print(f"  ✓ engagement_speed_score")

        if 'max_days_between_quotes' in df.columns:
            df['engagement_consistency'] = 1 / (df['max_days_between_quotes'] - df['avg_days_between_quotes'] + 1)
            efficiency_features['engagement_consistency'] = '1/(max_days - avg_days + 1)'
            print(f"  ✓ engagement_consistency")

    # 2. Decision complexity features
    if 'solution_complexity_score' in df.columns and 'equipment_variety_count' in df.columns:
        df['decision_complexity'] = df['solution_complexity_score'] * df['equipment_variety_count']
        efficiency_features['decision_complexity'] = 'complexity × variety'
        print(f"  ✓ decision_complexity")

    if 'brand_switch_rate' in df.columns:
        df['brand_consistency'] = 1 - df['brand_switch_rate']
        efficiency_features['brand_consistency'] = '1 - brand_switch_rate'
        print(f"  ✓ brand_consistency")

    # 3. Temporal patterns
    if 'summer_equipment_ratio' in df.columns and 'winter_equipment_ratio' in df.columns:
        df['seasonal_engagement_ratio'] = df['summer_equipment_ratio'] / (df['winter_equipment_ratio'] + 0.01)
        efficiency_features['seasonal_engagement_ratio'] = 'summer_ratio / (winter_ratio + 0.01)'
        print(f"  ✓ seasonal_engagement_ratio")

    return df, efficiency_features