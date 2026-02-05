
def create_efficiency_interaction_features(df, y_target=None):
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

    # 1. Decision Efficiency Features (Core Problem: short cycles penalized)
    if 'total_quotes' in df.columns and 'avg_days_between_quotes' in df.columns:
        # Inverse relationship - efficient = high quotes, low days
        df['decision_efficiency_score'] = df['total_quotes'] / (df['avg_days_between_quotes'] + 1)
        efficiency_features['decision_efficiency_score'] = 'quotes / days (efficiency)'

        # Short cycle indicator - directly addresses the core problem
        df['is_short_cycle'] = (df['avg_days_between_quotes'] < 7).astype(int)
        efficiency_features['is_short_cycle'] = 'avg_days < 7 (short cycle)'

        # Engagement intensity - rewards quick engagement
        df['engagement_intensity'] = df['total_quotes'] * (1 / (df['avg_days_between_quotes'] + 1))
        efficiency_features['engagement_intensity'] = 'quotes × (1/days)'

        # Very short cycle indicator (super efficient)
        df['is_very_short_cycle'] = (df['avg_days_between_quotes'] < 3).astype(int)
        efficiency_features['is_very_short_cycle'] = 'avg_days < 3 (very short)'

    # 2. Solution Clarity Features (Core Problem: simple solutions penalized)
    if 'equipment_variety_count' in df.columns and 'solution_complexity_score' in df.columns:
        # Simple solution indicator
        df['solution_focus_score'] = 1 / (df['equipment_variety_count'] * df['solution_complexity_score'] + 1)
        efficiency_features['solution_focus_score'] = '1/(variety × complexity)'

        # Efficient vs indecisive indicator
        df['is_focused_shopper'] = ((df['equipment_variety_count'] < 3) &
                                    (df['solution_complexity_score'] < 3)).astype(int)
        efficiency_features['is_focused_shopper'] = 'low variety & complexity'

        # Simple solution seeker
        df['is_simple_solution'] = ((df['equipment_variety_count'] == 1) &
                                    (df['solution_complexity_score'] < 2)).astype(int)
        efficiency_features['is_simple_solution'] = 'single equipment & simple'

    # 3. Price Discovery Efficiency
    if 'price_range' in df.columns and 'total_quotes' in df.columns:
        # Efficient price discovery = narrow range with few quotes
        df['price_discovery_efficiency'] = 1 / (df['price_range'] * df['total_quotes'] + 1)
        efficiency_features['price_discovery_efficiency'] = '1/(range × quotes)'

        # Budget certainty (inverse of price discovery)
        if 'max_price' in df.columns:
            df['budget_certainty'] = 1 - (df['price_range'] / (df['max_price'] + 1))
            efficiency_features['budget_certainty'] = '1 - range/max_price'

    # 4. Brand Loyalty Efficiency
    if 'brand_loyalty_index' in df.columns:
        # High loyalty = efficient decision making
        df['brand_efficiency'] = df['brand_loyalty_index']
        efficiency_features['brand_efficiency'] = 'brand_loyalty_index'

        if 'brand_switch_rate' in df.columns:
            df['brand_decision_efficiency'] = df['brand_loyalty_index'] * (1 - df['brand_switch_rate'])
            efficiency_features['brand_decision_efficiency'] = 'loyalty × (1 - switch_rate)'

    # 5. Historical Pattern Efficiency
    if 'total_historical_quotes' in df.columns and 'total_quotes' in df.columns:
        # Learning efficiency - current vs historical
        df['learning_efficiency'] = df['total_quotes'] / (df['total_historical_quotes'] + 1)
        efficiency_features['learning_efficiency'] = 'current_quotes / historical_quotes'

        # First-time vs repeat efficiency
        df['is_experienced_shopper'] = (df['total_historical_quotes'] > 0).astype(int)
        efficiency_features['is_experienced_shopper'] = 'has historical quotes'

    # 6. Intent Clarity Features (from your roadmap)
    if 'equipment_variety_count' in df.columns:
        df['equipment_consistency_score'] = 1 / (df['equipment_variety_count'] + 1)
        efficiency_features['equipment_consistency_score'] = '1/(equipment_variety + 1)'

    if 'min_price' in df.columns and 'max_price' in df.columns:
        df['price_discovery_range'] = df['max_price'] - df['min_price']
        efficiency_features['price_discovery_range'] = 'max_price - min_price'

    print(f"\n✅ Created {len(efficiency_features)} efficiency-focused features:")
    for name, desc in list(efficiency_features.items())[:15]:  # Show first 15
        print(f"  • {name}: {desc}")

    if len(efficiency_features) > 15:
        print(f"  ... and {len(efficiency_features) - 15} more features")

    # Quick validation if target is provided
    if y_target is not None:
        print("\n📊 QUICK CORRELATION WITH TARGET:")
        # Get a few key features to check
        key_features = ['decision_efficiency_score', 'is_short_cycle',
                        'solution_focus_score', 'engagement_intensity']

        for feat in key_features:
            if feat in df.columns:
                try:
                    corr = df[feat].corr(y_target)
                    print(f"  • {feat}: correlation = {corr:.3f}")
                except:
                    print(f"  • {feat}: could not compute correlation")
    else:
        print("\n📊 FEATURE STATISTICS (first 5 new features):")
        new_feature_names = list(efficiency_features.keys())[:5]
        for feat in new_feature_names:
            if feat in df.columns:
                print(f"  • {feat}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}")

    print(f"\n📈 Total features in dataset: {len(df.columns)}")
    print(f"🎯 New efficiency features: {len(efficiency_features)}")

    return df, efficiency_features
