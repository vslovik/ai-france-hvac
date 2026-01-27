import pandas as pd
import numpy as np


def create_solution_complexity_features(df):
    """
    VECTORIZED VERSION with chronological sorting for accuracy
    (LEAKAGE-FREE for predicting at last quote)
    """
    print("=" * 80)
    print("CREATING SOLUTION COMPLEXITY FEATURES (VECTORIZED)")
    print("=" * 80)

    # Check for equipment data
    equipment_col = 'regroup_famille_equipement_produit'
    if equipment_col not in df.columns:
        equipment_col = 'famille_equipement_produit'
        if equipment_col not in df.columns:
            print("  No equipment data available")
            return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing solution complexity for {df['numero_compte'].nunique():,} customers")

    # ========== CRITICAL: CHRONOLOGICAL SORTING ==========
    print("🔍 Ensuring chronological order...")
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)
        print("✅ Data sorted chronologically by customer and date")
    else:
        df = df.sort_values(['numero_compte']).reset_index(drop=True)
        print("⚠️  No date column - sorted by customer only")

    # Define solution architecture
    solution_components = {
        'heating_systems': ['BOILER_GAS', 'Chaudière', 'BOILER_OIL', 'HEAT_PUMP',
                            'CONDENSING_BOILER', 'STOVE', 'Poêle'],
        'cooling_systems': ['AIR_CONDITIONER', 'Climatisation', 'VRF_SYSTEM'],
        'hot_water': ['WATER_HEATER', 'CHAUFFE_EAU', 'BALLOON'],
        'ventilation': ['VENTILATION', 'VMC', 'AIR_EXCHANGER'],
        'renewable': ['SOLAR_THERMAL', 'GEOTHERMAL', 'PHOTOVOLTAIC'],
        'control': ['SMART_CONTROL', 'DOMOTIC', 'THERMOSTAT'],
        'plumbing': ['Plomberie Sanitaire', 'SANITARY', 'BATHROOM']
    }

    system_weights = {
        'heating_systems': 3.0,
        'cooling_systems': 2.5,
        'hot_water': 2.0,
        'ventilation': 1.5,
        'renewable': 3.5,
        'control': 2.0,
        'plumbing': 1.5
    }

    efficiency_indicators = [
        'CONDENS', 'INVERTER', 'A+++', 'A++', 'A+', 'ECO', 'EFFICIENT',
        'LOW_CONSUMPTION', 'ENERGY_SAVER', 'HEAT_PUMP', 'GEOTHERMAL'
    ]

    high_efficiency_equipment = ['HEAT_PUMP', 'GEOTHERMAL', 'SOLAR_THERMAL', 'CONDENSING_BOILER']

    # ========== VECTORIZED DATA PREPARATION ==========
    print("\n📊 Preparing data for vectorized processing...")

    # Create working DataFrame
    df_work = df[['numero_compte', equipment_col]].copy()
    df_work[equipment_col] = df_work[equipment_col].astype(str)

    # Filter customers with equipment data
    has_equipment = df_work[equipment_col].notna() & (df_work[equipment_col] != 'nan')
    df_equipment = df_work[has_equipment].copy()

    if len(df_equipment) == 0:
        print("⚠️  No equipment data found")
        return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing {len(df_equipment):,} equipment records")

    # ========== VECTORIZED SYSTEM TYPE MAPPING ==========
    print("🔧 Creating system type indicators...")

    # Create indicator columns for each system type
    system_dfs = []
    for sys_type, equipment_list in solution_components.items():
        # Create boolean column indicating if equipment belongs to this system
        sys_col = f'has_{sys_type}'
        df_equipment[sys_col] = df_equipment[equipment_col].isin(equipment_list).astype(int)
        system_dfs.append(sys_col)

    # ========== GROUP-BY AGGREGATIONS (VECTORIZED) ==========
    print("📈 Computing aggregated features...")

    # Group 1: System presence aggregations
    system_presence = df_equipment.groupby('numero_compte')[system_dfs].max().reset_index()

    # Count systems present
    system_presence['multi_system_count'] = system_presence[system_dfs].sum(axis=1)
    system_presence['has_multiple_systems'] = (system_presence['multi_system_count'] > 1).astype(int)

    # Core vs ancillary systems
    core_systems = ['has_heating_systems', 'has_cooling_systems', 'has_hot_water']
    ancillary_systems = ['has_ventilation', 'has_renewable', 'has_control', 'has_plumbing']

    system_presence['core_systems_count'] = system_presence[core_systems].sum(axis=1)
    system_presence['ancillary_systems_count'] = system_presence[ancillary_systems].sum(axis=1)
    system_presence['core_to_ancillary_ratio'] = system_presence['core_systems_count'] / system_presence[
        'ancillary_systems_count'].replace(0, 1)

    # ========== SOLUTION COMPLEXITY SCORE (VECTORIZED) ==========
    print("⚡ Calculating complexity scores...")

    complexity_score = 0
    for sys_type in solution_components.keys():
        col = f'has_{sys_type}'
        if col in system_presence.columns:
            weight = system_weights.get(sys_type, 1.0)
            complexity_score += system_presence[col] * weight

    # Integration bonus
    integration_bonus = 0.5 * np.maximum(system_presence['multi_system_count'] - 1, 0)
    system_presence['solution_complexity_score'] = complexity_score + integration_bonus

    # Normalized complexity
    max_possible = sum(system_weights.values()) + (0.5 * (len(system_weights) - 1))
    system_presence['normalized_complexity'] = system_presence['solution_complexity_score'] / max_possible

    # ========== COMPLETE SOLUTION ASSESSMENT (VECTORIZED) ==========
    heating_components = (
            system_presence['has_heating_systems'].fillna(0) +
            system_presence['has_hot_water'].fillna(0) +
            system_presence['has_control'].fillna(0)
    )

    system_presence['has_complete_heating_solution'] = (heating_components >= 2).astype(int)
    system_presence['heating_solution_completeness'] = heating_components / 3

    # ========== ENERGY EFFICIENCY SCORE (VECTORIZED) ==========
    print("🌱 Calculating energy efficiency...")

    # Create uppercase version for string matching
    df_equipment['equipment_upper'] = df_equipment[equipment_col].str.upper()

    # Check for efficiency indicators
    efficiency_matches = pd.DataFrame()
    for indicator in efficiency_indicators:
        efficiency_matches[indicator] = df_equipment['equipment_upper'].str.contains(indicator, regex=False,
                                                                                     na=False).astype(int)

    # Sum matches per quote
    df_equipment['efficiency_indicator_count'] = efficiency_matches.sum(axis=1)

    # Check for high efficiency equipment
    df_equipment['is_high_efficiency'] = df_equipment[equipment_col].isin(high_efficiency_equipment).astype(int)

    # Aggregate to customer level
    efficiency_agg = df_equipment.groupby('numero_compte').agg(
        efficiency_score=('efficiency_indicator_count', 'sum'),
        high_efficiency_count=('is_high_efficiency', 'sum')
    ).reset_index()

    # Total efficiency score
    efficiency_agg['energy_efficiency_score'] = efficiency_agg['efficiency_score'] + (
                efficiency_agg['high_efficiency_count'] * 2)
    efficiency_agg['unique_efficiency_indicators'] = efficiency_agg['efficiency_score']
    efficiency_agg['has_high_efficiency_system'] = (efficiency_agg['energy_efficiency_score'] >= 2).astype(int)

    # ========== SYSTEM INTEGRATION LEVEL (VECTORIZED) ==========
    print("🔗 Computing system integration...")

    integration_factors = []

    # Factor 1: Heating + Control
    integration_factors.append(
        (system_presence['has_heating_systems'] & system_presence['has_control']) * 0.8
    )

    # Factor 2: Multi-season capability
    integration_factors.append(
        (system_presence['has_heating_systems'] & system_presence['has_cooling_systems']) * 0.9
    )

    # Factor 3: Renewable integration
    renewable_integration = (
            system_presence['has_renewable'] &
            (system_presence['has_heating_systems'] | system_presence['has_hot_water'])
    )
    integration_factors.append(renewable_integration * 1.0)

    # Factor 4: Ventilation with heating/cooling
    ventilation_integration = (
            system_presence['has_ventilation'] &
            (system_presence['has_heating_systems'] | system_presence['has_cooling_systems'])
    )
    integration_factors.append(ventilation_integration * 0.6)

    # Calculate mean integration level
    integration_matrix = np.column_stack(integration_factors)
    valid_factors = integration_matrix > 0
    integration_scores = []

    for i in range(len(integration_matrix)):
        valid_values = integration_matrix[i][valid_factors[i]]
        if len(valid_values) > 0:
            integration_scores.append(np.mean(valid_values))
        else:
            integration_scores.append(0)

    system_presence['system_integration_level'] = integration_scores
    system_presence['integration_factors_count'] = valid_factors.sum(axis=1)

    # ========== PRIMARY SYSTEM DOMINANCE (VECTORIZED) ==========
    print("🏆 Determining primary systems...")

    # Get most frequent equipment per customer
    equipment_counts = df_equipment.groupby(['numero_compte', equipment_col]).size().reset_index(name='count')
    primary_equipment = equipment_counts.sort_values(['numero_compte', 'count'], ascending=[True, False])
    primary_equipment = primary_equipment.drop_duplicates('numero_compte', keep='first')

    # Calculate dominance ratio
    total_counts = df_equipment.groupby('numero_compte').size().reset_index(name='total_count')
    dominance = pd.merge(primary_equipment, total_counts, on='numero_compte')
    dominance['primary_system_dominance'] = dominance['count'] / dominance['total_count']

    # Map primary equipment to system type
    def map_to_system_type(equipment):
        for sys_type, equipment_list in solution_components.items():
            if equipment in equipment_list:
                return sys_type
        return 'other'

    dominance['primary_system_type'] = dominance[equipment_col].apply(map_to_system_type)

    # ========== MERGE ALL FEATURES ==========
    print("🔄 Merging all features...")

    result = system_presence.copy()
    result = pd.merge(result, efficiency_agg[['numero_compte', 'energy_efficiency_score',
                                              'unique_efficiency_indicators', 'has_high_efficiency_system']],
                      on='numero_compte', how='left')
    result = pd.merge(result, dominance[['numero_compte', 'primary_system_dominance', 'primary_system_type']],
                      on='numero_compte', how='left')

    # ========== SOLUTION SOPHISTICATION TIER (VECTORIZED) ==========
    print("📊 Calculating sophistication tiers...")

    tier_score = 0
    tier_score += (result['solution_complexity_score'] >= 5).astype(int) * 2
    tier_score += ((result['solution_complexity_score'] >= 3) & (result['solution_complexity_score'] < 5)).astype(int)
    tier_score += result['has_high_efficiency_system'].fillna(0)
    tier_score += (result['system_integration_level'] >= 0.5).astype(int)
    tier_score += result['has_complete_heating_solution']

    result['sophistication_score'] = tier_score

    # Assign tier categories
    result['solution_sophistication_tier'] = np.select(
        [
            tier_score >= 4,
            tier_score >= 2
        ],
        ['advanced', 'intermediate'],
        default='basic'
    )

    # ========== FUTURE-PROOFING INDICATOR (VECTORIZED) ==========
    future_proofing = 0
    future_proofing += result['has_renewable']
    future_proofing += result['has_high_efficiency_system'].fillna(0)
    future_proofing += result['has_control']
    future_proofing += (result['multi_system_count'] > 1).astype(int)

    result['future_proofing_score'] = future_proofing / 4

    # ========== ADD SOLUTION DATA AVAILABILITY FLAG ==========
    all_customers = pd.DataFrame({'numero_compte': df['numero_compte'].unique()})
    result = pd.merge(all_customers, result, on='numero_compte', how='left')

    # Fill missing values for customers without equipment data
    result['solution_data_available'] = result['multi_system_count'].notna().astype(int)
    result = result.fillna({
        'multi_system_count': 0,
        'solution_complexity_score': 0,
        'has_complete_heating_solution': 0,
        'energy_efficiency_score': 0,
        'system_integration_level': 0,
        'primary_system_dominance': 0,
        'sophistication_score': 0,
        'future_proofing_score': 0,
        'primary_system_type': 'none'
    })

    # ========== FINAL REPORT ==========
    print(f"\n✅ Created {len(result.columns) - 1} solution complexity features")
    print(f"   Samples: {len(result):,} customers")

    # Show key features
    if len(result) > 0:
        key_features = [
            'multi_system_count', 'solution_complexity_score',
            'has_complete_heating_solution', 'energy_efficiency_score',
            'system_integration_level', 'primary_system_dominance',
            'sophistication_score', 'future_proofing_score'
        ]

        print("\n📊 SOLUTION COMPLEXITY FEATURES SUMMARY:")
        print("-" * 60)
        for feat in key_features:
            if feat in result.columns:
                mean_val = result[feat].mean()
                std_val = result[feat].std()
                print(f"{feat:35} : mean={mean_val:.3f}, std={std_val:.3f}")

    return result
