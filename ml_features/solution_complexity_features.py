import pandas as pd
import numpy as np


def create_solution_complexity_features(df):
    """
    PILLAR 2.2: Solution Complexity Assessment

    Creates features that capture solution sophistication:
    1. Multi-system consideration and integration
    2. Primary vs secondary system analysis
    3. Energy efficiency and technology level
    4. System completeness assessment
    """
    print("=" * 80)
    print("CREATING SOLUTION COMPLEXITY FEATURES")
    print("=" * 80)

    # Check for equipment data
    equipment_col = 'regroup_famille_equipement_produit'
    if equipment_col not in df.columns:
        equipment_col = 'famille_equipement_produit'
        if equipment_col not in df.columns:
            print("  No equipment data available")
            return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing solution complexity for {df['numero_compte'].nunique():,} customers")

    # Sort for sequence analysis
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()
    else:
        df = df.sort_values(['numero_compte']).copy()

    # Define solution architecture
    solution_components = {
        # Core systems
        'heating_systems': ['BOILER_GAS', 'Chaudière', 'BOILER_OIL', 'HEAT_PUMP',
                            'CONDENSING_BOILER', 'STOVE', 'Poêle'],
        'cooling_systems': ['AIR_CONDITIONER', 'Climatisation', 'VRF_SYSTEM'],
        'hot_water': ['WATER_HEATER', 'CHAUFFE_EAU', 'BALLOON'],
        'ventilation': ['VENTILATION', 'VMC', 'AIR_EXCHANGER'],

        # Advanced/ancillary systems
        'renewable': ['SOLAR_THERMAL', 'GEOTHERMAL', 'PHOTOVOLTAIC'],
        'control': ['SMART_CONTROL', 'DOMOTIC', 'THERMOSTAT'],
        'plumbing': ['Plomberie Sanitaire', 'SANITARY', 'BATHROOM']
    }

    # System complexity weights
    system_weights = {
        'heating_systems': 3.0,  # Core, high complexity
        'cooling_systems': 2.5,  # Core, moderate complexity
        'hot_water': 2.0,  # Essential, moderate complexity
        'ventilation': 1.5,  # Supplemental, lower complexity
        'renewable': 3.5,  # Advanced, high complexity
        'control': 2.0,  # Enhancement, moderate complexity
        'plumbing': 1.5  # Basic, lower complexity
    }

    # Energy efficiency indicators
    efficiency_indicators = [
        'CONDENS', 'INVERTER', 'A+++', 'A++', 'A+', 'ECO', 'EFFICIENT',
        'LOW_CONSUMPTION', 'ENERGY_SAVER', 'HEAT_PUMP', 'GEOTHERMAL'
    ]

    complexity_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        equipment_data = customer_data[equipment_col].dropna()

        if len(equipment_data) == 0:
            # Default values
            features.update({
                'solution_data_available': 0,
                'multi_system_count': 0,
                'solution_complexity_score': 0,
                'has_complete_heating_solution': 0,
                'energy_efficiency_score': 0,
                'system_integration_level': 0,
                'primary_system_dominance': 0,
                'ancillary_systems_count': 0
            })
        else:
            features['solution_data_available'] = 1

            # ========== FEATURE 1: MULTI-SYSTEM ANALYSIS ==========
            system_presence = {sys_type: 0 for sys_type in solution_components.keys()}
            unique_equipment = set(equipment_data)

            # Map equipment to system types
            for equip in unique_equipment:
                for sys_type, sys_equipment in solution_components.items():
                    if equip in sys_equipment:
                        system_presence[sys_type] = 1
                        break

            # Count and weight systems
            present_systems = [sys for sys, present in system_presence.items() if present == 1]
            features['multi_system_count'] = len(present_systems)
            features['has_multiple_systems'] = 1 if len(present_systems) > 1 else 0

            # Core vs ancillary systems
            core_systems = ['heating_systems', 'cooling_systems', 'hot_water']
            ancillary_systems = ['ventilation', 'renewable', 'control', 'plumbing']

            core_count = sum(1 for sys in core_systems if system_presence.get(sys, 0) == 1)
            ancillary_count = sum(1 for sys in ancillary_systems if system_presence.get(sys, 0) == 1)

            features['core_systems_count'] = core_count
            features['ancillary_systems_count'] = ancillary_count
            features['core_to_ancillary_ratio'] = core_count / max(ancillary_count, 1)

            # ========== FEATURE 2: SOLUTION COMPLEXITY SCORE ==========
            complexity_score = 0
            for sys_type in present_systems:
                complexity_score += system_weights.get(sys_type, 1.0)

            # Adjust for system integration (more systems = more complex integration)
            if len(present_systems) > 1:
                integration_bonus = 0.5 * (len(present_systems) - 1)
                complexity_score += integration_bonus

            features['solution_complexity_score'] = complexity_score

            # Normalized complexity (0-1 scale)
            max_possible = sum(system_weights.values()) + (0.5 * (len(system_weights) - 1))
            features['normalized_complexity'] = complexity_score / max_possible if max_possible > 0 else 0

            # ========== FEATURE 3: COMPLETE SOLUTION ASSESSMENT ==========
            # A "complete" heating solution might include multiple components
            heating_components = []
            if system_presence.get('heating_systems', 0) == 1:
                heating_components.append('primary_heating')
            if system_presence.get('hot_water', 0) == 1:
                heating_components.append('hot_water')
            if system_presence.get('control', 0) == 1:
                heating_components.append('smart_control')

            features['has_complete_heating_solution'] = 1 if len(heating_components) >= 2 else 0
            features['heating_solution_completeness'] = len(heating_components) / 3  # 3 possible components

            # ========== FEATURE 4: ENERGY EFFICIENCY SCORE ==========
            efficiency_score = 0
            efficiency_indicators_found = []

            # Check equipment names for efficiency indicators
            for equip in equipment_data:
                equip_upper = str(equip).upper()
                for indicator in efficiency_indicators:
                    if indicator in equip_upper:
                        efficiency_indicators_found.append(indicator)
                        efficiency_score += 1

            # Check for high-efficiency equipment types
            high_efficiency_equipment = ['HEAT_PUMP', 'GEOTHERMAL', 'SOLAR_THERMAL', 'CONDENSING_BOILER']
            for equip in unique_equipment:
                if equip in high_efficiency_equipment:
                    efficiency_score += 2  # Extra weight for high-efficiency systems

            features['energy_efficiency_score'] = efficiency_score
            features['unique_efficiency_indicators'] = len(set(efficiency_indicators_found))
            features['has_high_efficiency_system'] = 1 if efficiency_score >= 2 else 0

            # ========== FEATURE 5: SYSTEM INTEGRATION LEVEL ==========
            # Measures how well systems work together
            integration_factors = []

            # Factor 1: Heating + Control integration
            if system_presence.get('heating_systems', 0) == 1 and system_presence.get('control', 0) == 1:
                integration_factors.append(0.8)

            # Factor 2: Multi-season capability
            if system_presence.get('heating_systems', 0) == 1 and system_presence.get('cooling_systems', 0) == 1:
                integration_factors.append(0.9)  # Year-round comfort

            # Factor 3: Renewable integration
            if system_presence.get('renewable', 0) == 1 and (system_presence.get('heating_systems', 0) == 1 or
                                                             system_presence.get('hot_water', 0) == 1):
                integration_factors.append(1.0)  # Advanced integration

            # Factor 4: Ventilation with heating/cooling
            if system_presence.get('ventilation', 0) == 1 and (system_presence.get('heating_systems', 0) == 1 or
                                                               system_presence.get('cooling_systems', 0) == 1):
                integration_factors.append(0.6)

            features['system_integration_level'] = np.mean(integration_factors) if integration_factors else 0
            features['integration_factors_count'] = len(integration_factors)

            # ========== FEATURE 6: PRIMARY SYSTEM DOMINANCE ==========
            # Does one system type dominate their considerations?
            equipment_counts = equipment_data.value_counts()
            if len(equipment_counts) > 0:
                primary_count = equipment_counts.iloc[0]
                total_count = len(equipment_data)
                features['primary_system_dominance'] = primary_count / total_count

                # Identify primary system type
                primary_equipment = equipment_counts.index[0]
                primary_system_type = 'other'
                for sys_type, sys_equipment in solution_components.items():
                    if primary_equipment in sys_equipment:
                        primary_system_type = sys_type
                        break
                features['primary_system_type'] = primary_system_type
            else:
                features['primary_system_dominance'] = 0
                features['primary_system_type'] = 'none'

            # ========== FEATURE 7: SOLUTION SOPHISTICATION TIER ==========
            # Classify customers into sophistication tiers
            tier_score = 0

            # Tier components
            if features['solution_complexity_score'] >= 5:
                tier_score += 2  # High complexity
            elif features['solution_complexity_score'] >= 3:
                tier_score += 1  # Medium complexity

            if features['has_high_efficiency_system'] == 1:
                tier_score += 1  # Energy efficient

            if features['system_integration_level'] >= 0.5:
                tier_score += 1  # Good integration

            if features['has_complete_heating_solution'] == 1:
                tier_score += 1  # Complete solution

            # Assign tier
            if tier_score >= 4:
                features['solution_sophistication_tier'] = 'advanced'
            elif tier_score >= 2:
                features['solution_sophistication_tier'] = 'intermediate'
            else:
                features['solution_sophistication_tier'] = 'basic'

            features['sophistication_score'] = tier_score

            # ========== FEATURE 8: FUTURE-PROOFING INDICATOR ==========
            # Does their solution consider future needs/trends?
            future_proofing_indicators = []

            if system_presence.get('renewable', 0) == 1:
                future_proofing_indicators.append(1)  # Renewable energy

            if features['has_high_efficiency_system'] == 1:
                future_proofing_indicators.append(1)  # Energy efficiency

            if system_presence.get('control', 0) == 1:
                future_proofing_indicators.append(1)  # Smart controls

            if len(present_systems) > 1:
                future_proofing_indicators.append(1)  # Integrated systems

            features['future_proofing_score'] = len(future_proofing_indicators) / 4  # 4 possible indicators

        complexity_features.append(features)

    # Convert to DataFrame
    complexity_features_df = pd.DataFrame(complexity_features)

    # Report statistics
    print(f"\n✅ Created {len(complexity_features_df.columns) - 1} solution complexity features")
    print(f"   Samples: {len(complexity_features_df):,}")

    # Show key features
    if len(complexity_features_df) > 0:
        key_features = [
            'multi_system_count', 'solution_complexity_score',
            'has_complete_heating_solution', 'energy_efficiency_score',
            'system_integration_level', 'primary_system_dominance',
            'sophistication_score', 'future_proofing_score'
        ]

        print("\n📊 SOLUTION COMPLEXITY FEATURES SUMMARY:")
        print("-" * 60)
        for feat in key_features:
            if feat in complexity_features_df.columns:
                mean_val = complexity_features_df[feat].mean()
                std_val = complexity_features_df[feat].std()
                print(f"{feat:35} : mean={mean_val:.3f}, std={std_val:.3f}")

    return complexity_features_df
