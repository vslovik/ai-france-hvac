def create_catboost_interaction_features(df):
    """
    Complete Phase B feature engineering combining:
    1. CatBoost interaction insights
    2. Efficiency-focused features (from roadmap)
    3. Intent clarity features
    """
    print("=" * 80)
    print("🔧 COMPLETE PHASE B FEATURE ENGINEERING")
    print("=" * 80)

    df_phase_b = df.copy()
    phase_b_features = {}

    # PART 1: CatBoost Interaction Insights (from SHAP analysis)
    print("\n1️⃣ CATBOOST INTERACTION FEATURES")
    print("-" * 40)

    # Strong interactions found: engagement_days × std_recent_product_variety
    if 'engagement_days' in df_phase_b.columns and 'std_recent_product_variety' in df_phase_b.columns:
        df_phase_b['engagement_product_interaction'] = df_phase_b['engagement_days'] * df_phase_b[
            'std_recent_product_variety']
        phase_b_features['engagement_product_interaction'] = 'engagement_days × product_variety'

    # Strong interaction: total_historical_quotes × price_trajectory
    if 'total_historical_quotes' in df_phase_b.columns and 'price_trajectory' in df_phase_b.columns:
        df_phase_b['historical_price_interaction'] = df_phase_b['total_historical_quotes'] * df_phase_b[
            'price_trajectory']
        phase_b_features['historical_price_interaction'] = 'historical_quotes × price_trajectory'

    # Quote consistency
    if 'quote_count' in df_phase_b.columns and 'total_historical_quotes' in df_phase_b.columns:
        df_phase_b['quote_consistency_score'] = df_phase_b['quote_count'] / (df_phase_b['total_historical_quotes'] + 1)
        phase_b_features['quote_consistency_score'] = 'quote_count / historical_quotes'

    # Price sensitivity
    if 'quote_count' in df_phase_b.columns and 'price_trajectory' in df_phase_b.columns:
        df_phase_b['quote_price_sensitivity'] = df_phase_b['quote_count'] * abs(df_phase_b['price_trajectory'])
        phase_b_features['quote_price_sensitivity'] = 'quote_count × |price_trajectory|'

    # PART 2: Efficiency-Focused Features (addressing core problem)
    print("\n2️⃣ EFFICIENCY-FOCUSED FEATURES")
    print("-" * 40)

    # Decision efficiency (short cycles)
    if 'total_quotes' in df_phase_b.columns and 'avg_days_between_quotes' in df_phase_b.columns:
        df_phase_b['decision_efficiency_score'] = df_phase_b['total_quotes'] / (
                    df_phase_b['avg_days_between_quotes'] + 1)
        phase_b_features['decision_efficiency_score'] = 'quotes / days (efficiency)'

        df_phase_b['is_short_cycle'] = (df_phase_b['avg_days_between_quotes'] < 7).astype(int)
        phase_b_features['is_short_cycle'] = 'avg_days < 7 (short cycle)'

        df_phase_b['engagement_velocity'] = df_phase_b['total_quotes'] / (df_phase_b['avg_days_between_quotes'] + 1)
        phase_b_features['engagement_velocity'] = 'total_quotes / avg_days'

    # Solution clarity (simple solutions)
    if 'equipment_variety_count' in df_phase_b.columns and 'solution_complexity_score' in df_phase_b.columns:
        df_phase_b['solution_focus_score'] = 1 / (
                    df_phase_b['equipment_variety_count'] * df_phase_b['solution_complexity_score'] + 1)
        phase_b_features['solution_focus_score'] = '1/(variety × complexity)'

        df_phase_b['is_focused_shopper'] = ((df_phase_b['equipment_variety_count'] < 3) &
                                            (df_phase_b['solution_complexity_score'] < 3)).astype(int)
        phase_b_features['is_focused_shopper'] = 'low variety & complexity'

    # PART 3: Intent Clarity Features (from your roadmap)
    print("\n3️⃣ INTENT CLARITY FEATURES (From Roadmap)")
    print("-" * 40)

    # Equipment consistency
    if 'equipment_variety_count' in df_phase_b.columns:
        df_phase_b['equipment_consistency_score'] = 1 / (df_phase_b['equipment_variety_count'] + 1)
        phase_b_features['equipment_consistency_score'] = '1/(equipment_variety + 1)'

    # Price discovery
    if 'min_price' in df_phase_b.columns and 'max_price' in df_phase_b.columns:
        df_phase_b['price_discovery_range'] = df_phase_b['max_price'] - df_phase_b['min_price']
        phase_b_features['price_discovery_range'] = 'max_price - min_price'

        df_phase_b['price_consistency'] = 1 - (df_phase_b['price_discovery_range'] / (df_phase_b['max_price'] + 1))
        phase_b_features['price_consistency'] = '1 - price_range/max_price'

    # Brand switching
    if 'brand_switch_count' in df_phase_b.columns:
        df_phase_b['brand_switch_count'] = df_phase_b['brand_switch_count']
        phase_b_features['brand_switch_count'] = 'brand_switch_count'

    # PART 4: Budget Certainty
    print("\n4️⃣ BUDGET CERTAINTY FEATURES")
    print("-" * 40)

    if 'price_range' in df_phase_b.columns and 'total_quotes' in df_phase_b.columns:
        df_phase_b['price_discovery_efficiency'] = 1 / (df_phase_b['price_range'] * df_phase_b['total_quotes'] + 1)
        phase_b_features['price_discovery_efficiency'] = '1/(range × quotes)'

    # Summary
    print(f"\n✅ PHASE B COMPLETE")
    print(f"   Total new features created: {len(phase_b_features)}")
    print(f"   Total features in dataset: {len(df_phase_b.columns)}")

    print(f"\n📋 FEATURE CATEGORIES:")
    categories = {
        'CatBoost Interactions': ['engagement_product_interaction', 'historical_price_interaction',
                                  'quote_consistency_score', 'quote_price_sensitivity'],
        'Efficiency Focus': ['decision_efficiency_score', 'is_short_cycle', 'engagement_velocity',
                             'solution_focus_score', 'is_focused_shopper'],
        'Intent Clarity': ['equipment_consistency_score', 'price_discovery_range',
                           'price_consistency', 'brand_switch_count'],
        'Budget Certainty': ['price_discovery_efficiency']
    }

    for category, features in categories.items():
        actual_features = [f for f in features if f in phase_b_features]
        if actual_features:
            print(f"  • {category}: {len(actual_features)} features")
            for feat in actual_features[:3]:  # Show first 3
                print(f"    - {feat}")
            if len(actual_features) > 3:
                print(f"    ... and {len(actual_features) - 3} more")

    return df_phase_b, phase_b_features
