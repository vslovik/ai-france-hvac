
def create_simulation_discovery_features(df):
    """
    Add ONLY the interaction features discovered from your simulation POC.
    Uses existing customer-level columns created by other feature functions.
    """
    print("=" * 80)
    print("CREATING SIMULATION-DISCOVERY FEATURES")
    print("=" * 80)

    # ============================================
    # 1. Verify required columns exist (from other feature functions)
    # ============================================

    required = [
        'has_heat_pump', 'has_stove', 'has_boiler', 'has_ac',
        'is_single_quote', 'customer_quote_count'
    ]

    # Check which columns are already in the dataframe
    existing = [col for col in required if col in df.columns]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"⚠️ Missing columns (will skip dependent features): {missing}")

    # ============================================
    # 2. Cross-sell opportunities (from your simulations)
    # ============================================

    if 'has_heat_pump' in df.columns and 'has_stove' in df.columns:
        df['heat_pump_to_stove_opportunity'] = (
                (df['has_heat_pump'] == 1) &
                (df['has_stove'] == 0)
        ).astype(int)
        print(f"✅ Created 'heat_pump_to_stove_opportunity'")

    if 'has_boiler' in df.columns and 'has_ac' in df.columns:
        df['boiler_to_ac_opportunity'] = (
                (df['has_boiler'] == 1) &
                (df['has_ac'] == 0)
        ).astype(int)
        print(f"✅ Created 'boiler_to_ac_opportunity'")

    if 'has_stove' in df.columns and 'has_heat_pump' in df.columns:
        df['stove_to_heat_pump_opportunity'] = (
                (df['has_stove'] == 1) &
                (df['has_heat_pump'] == 0)
        ).astype(int)
        print(f"✅ Created 'stove_to_heat_pump_opportunity'")

    # ============================================
    # 3. Regional interaction features
    # ============================================

    # Check for region column
    if 'nom_region' in df.columns:
        COLD_REGIONS = [
            'Normandie', 'Hauts-de-France', 'Grand Est',
            'Bourgogne-Franche-Comté', 'Bretagne'
        ]

        df['is_cold_region'] = df['nom_region'].isin(COLD_REGIONS).astype(int)

        if 'has_heat_pump' in df.columns:
            df['cold_region_heat_pump'] = df['is_cold_region'] * df['has_heat_pump']
            print(f"✅ Created 'cold_region_heat_pump'")

        # The winning cross-sell: Heat pump in cold region needs stove
        if 'has_heat_pump' in df.columns and 'has_stove' in df.columns:
            df['cold_heat_pump_to_stove'] = (
                    df['is_cold_region'] &
                    df['has_heat_pump'] &
                    (df['has_stove'] == 0)
            ).astype(int)
            print(f"✅ Created 'cold_heat_pump_to_stove'")

    # ============================================
    # 4. Engagement features (if not already present)
    # ============================================

    if 'customer_quote_count' in df.columns:
        df['follow_up_opportunity'] = (df['customer_quote_count'] == 1).astype(int)
        print(f"✅ Created 'follow_up_opportunity'")
    elif 'is_single_quote' in df.columns:
        df['follow_up_opportunity'] = df['is_single_quote']
        print(f"✅ Created 'follow_up_opportunity' from existing 'is_single_quote'")

    # ============================================
    # 5. Summary
    # ============================================

    new_features = [col for col in df.columns if col not in required + ['nom_region']]
    print(f"\n✅ Added {len(new_features)} simulation-discovery features")

    # Cold region heat pump owners get a boost to value_score
    if 'cold_region_heat_pump' in df.columns and 'value_score' in df.columns:
        df['value_score'] = df['value_score'] * (1 + df['cold_region_heat_pump'] * 0.126)

    # Heat pump to stove opportunity boosts price_x_sophistication
    if 'heat_pump_to_stove_opportunity' in df.columns and 'price_x_sophistication' in df.columns:
        df['price_x_sophistication'] = df['price_x_sophistication'] * (1 + df['heat_pump_to_stove_opportunity'] * 0.126)

    # Boiler to AC opportunity boosts avg_discount_pct
    if 'boiler_to_ac_opportunity' in df.columns and 'avg_discount_pct' in df.columns:
        df['avg_discount_pct'] = df['avg_discount_pct'] * (1 + df['boiler_to_ac_opportunity'] * 0.056)

    return df