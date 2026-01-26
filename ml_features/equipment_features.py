import pandas as pd
import numpy as np


def create_equipment_features(df):
    """
    PILLAR 2.1: Equipment Upgrade Path Features

    Creates features that capture equipment upgrade trajectories:
    1. Upgrade Trajectory Score: Movement from basic → advanced equipment
    2. Equipment Family Consistency: Loyalty to one equipment type
    3. Seasonal Equipment Mix: Balance between seasonal needs
    """
    print("=" * 80)
    print("CREATING EQUIPMENT UPGRADE PATH FEATURES")
    print("=" * 80)

    # Check for equipment data
    equipment_col = 'regroup_famille_equipement_produit'
    if equipment_col not in df.columns:
        print(f"⚠️ WARNING: '{equipment_col}' column not found")
        # Try alternative
        equipment_col = 'famille_equipement_produit'
        if equipment_col not in df.columns:
            print("  No equipment data available")
            return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing equipment upgrade data for {df['numero_compte'].nunique():,} customers")
    print(f"Total quotes with equipment info: {df[equipment_col].notna().sum():,}")

    # Ensure we have datetime for sequence analysis
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()
    else:
        df = df.sort_values(['numero_compte']).copy()
        print("  Note: No date column for temporal analysis, using quote order")

    # ========== DEFINE EQUIPMENT HIERARCHY & ATTRIBUTES ==========

    # Equipment complexity hierarchy (1=basic, 5=advanced)
    equipment_complexity = {
        # Basic equipment (simple, often supplemental)
        'STOVE': 1, 'Poêle': 1, 'CHEMINEE': 1, 'INSERT': 1,

        # Intermediate equipment (primary systems)
        'BOILER_GAS': 2, 'Chaudière': 2, 'BOILER_OIL': 2,
        'AIR_CONDITIONER': 2, 'Climatisation': 2,
        'Plomberie Sanitaire': 2, 'SANITARY': 2,

        # Advanced equipment (efficient, complex)
        'HEAT_PUMP': 3, 'Pompe à chaleur': 3,
        'CONDENSING_BOILER': 3, 'CHAUDIERE_CONDENSATION': 3,
        'VRF_SYSTEM': 3, 'SYSTEME_VRF': 3,

        # Very advanced (cutting edge)
        'GEOTHERMAL': 4, 'GEOTHERMIE': 4,
        'SOLAR_THERMAL': 4, 'SOLAIRE_THERMIQUE': 4,
        'HYBRID_SYSTEM': 4, 'SYSTEME_HYBRIDE': 4,

        # Default
        'OTHER': 1.5, 'AUTRES': 1.5
    }

    # Equipment seasonality
    equipment_seasonality = {
        # Winter-focused (heating)
        'STOVE': 'winter', 'Poêle': 'winter',
        'BOILER_GAS': 'winter', 'Chaudière': 'winter',
        'BOILER_OIL': 'winter', 'HEAT_PUMP': 'winter',
        'CONDENSING_BOILER': 'winter', 'GEOTHERMAL': 'winter',

        # Summer-focused (cooling)
        'AIR_CONDITIONER': 'summer', 'Climatisation': 'summer',
        'VRF_SYSTEM': 'summer',

        # Year-round
        'Plomberie Sanitaire': 'year_round', 'SANITARY': 'year_round',
        'SOLAR_THERMAL': 'year_round', 'HYBRID_SYSTEM': 'year_round',
        'OTHER': 'year_round', 'AUTRES': 'year_round'
    }

    # Equipment upgrade paths (from → to)
    upgrade_paths = {
        ('STOVE', 'BOILER_GAS'): 'basic_to_primary_heating',
        ('STOVE', 'HEAT_PUMP'): 'basic_to_advanced',
        ('BOILER_GAS', 'HEAT_PUMP'): 'primary_to_advanced',
        ('BOILER_GAS', 'CONDENSING_BOILER'): 'standard_to_efficient',
        ('AIR_CONDITIONER', 'HEAT_PUMP'): 'cooling_to_hybrid',
        ('BOILER_GAS', 'GEOTHERMAL'): 'conventional_to_renewable'
    }

    upgrade_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        equipment_sequence = customer_data[equipment_col].dropna()

        if len(equipment_sequence) == 0:
            # No equipment data
            features.update({
                'equipment_upgrade_data_available': 0,
                'equipment_family_consistency': 0,
                'upgrade_trajectory_score': 0,
                'seasonal_equipment_mix': 0,
                'has_upgrade_pattern': 0,
                'equipment_complexity_trend': 0,
                'equipment_variety_index': 0,
                'primary_season_focus': 'none',
                'equipment_maturity_level': 0
            })
        else:
            features['equipment_upgrade_data_available'] = 1

            # ========== FEATURE 1: EQUIPMENT FAMILY CONSISTENCY ==========
            unique_equipment = equipment_sequence.nunique()
            features['equipment_family_consistency'] = 1 if unique_equipment == 1 else 0
            features['equipment_variety_count'] = unique_equipment

            # Calculate variety index (0-1, higher = more variety)
            if len(equipment_sequence) > 1:
                max_variety = min(len(equipment_sequence), len(equipment_complexity))
                features['equipment_variety_index'] = (unique_equipment - 1) / (max_variety - 1)
            else:
                features['equipment_variety_index'] = 0  # Single quote = no variety

            # ========== FEATURE 2: UPGRADE TRAJECTORY SCORE ==========
            if len(equipment_sequence) > 1:
                # Convert equipment to complexity scores
                complexity_scores = []
                for equip in equipment_sequence:
                    score = equipment_complexity.get(equip, 1.5)  # Default to medium-basic
                    complexity_scores.append(score)

                # Calculate linear trend
                x = np.arange(len(complexity_scores))
                if len(set(complexity_scores)) > 1:  # Need variation for regression
                    slope, intercept = np.polyfit(x, complexity_scores, 1)
                    features['upgrade_trajectory_score'] = slope

                    # Additional trend metrics
                    features['equipment_complexity_trend'] = 1 if slope > 0.05 else (-1 if slope < -0.05 else 0)
                    features['final_complexity'] = complexity_scores[-1]
                    features['complexity_range'] = max(complexity_scores) - min(complexity_scores)
                else:
                    features['upgrade_trajectory_score'] = 0
                    features['equipment_complexity_trend'] = 0
                    features['final_complexity'] = complexity_scores[0] if complexity_scores else 0
                    features['complexity_range'] = 0

                # Check for specific upgrade patterns
                upgrade_detected = 0
                if len(equipment_sequence) >= 2:
                    first_eq = equipment_sequence.iloc[0]
                    last_eq = equipment_sequence.iloc[-1]

                    first_complexity = equipment_complexity.get(first_eq, 1.5)
                    last_complexity = equipment_complexity.get(last_eq, 1.5)

                    # Binary: Did they upgrade?
                    features['has_upgrade'] = 1 if last_complexity > first_complexity else 0
                    features['has_downgrade'] = 1 if last_complexity < first_complexity else 0

                    # Check specific upgrade paths
                    upgrade_key = (first_eq, last_eq)
                    if upgrade_key in upgrade_paths:
                        upgrade_detected = 1
                        features['specific_upgrade_type'] = upgrade_paths[upgrade_key]
                    else:
                        features['specific_upgrade_type'] = 'none'
                else:
                    features['has_upgrade'] = 0
                    features['has_downgrade'] = 0
                    features['specific_upgrade_type'] = 'none'

                features['has_upgrade_pattern'] = upgrade_detected

            else:
                # Single quote customers
                single_complexity = equipment_complexity.get(equipment_sequence.iloc[0], 1.5)
                features['upgrade_trajectory_score'] = 0
                features['equipment_complexity_trend'] = 0
                features['final_complexity'] = single_complexity
                features['complexity_range'] = 0
                features['has_upgrade'] = 0
                features['has_downgrade'] = 0
                features['specific_upgrade_type'] = 'none'
                features['has_upgrade_pattern'] = 0

            # ========== FEATURE 3: SEASONAL EQUIPMENT MIX ==========
            season_counts = {'winter': 0, 'summer': 0, 'year_round': 0}

            for equip in equipment_sequence:
                season = equipment_seasonality.get(equip, 'year_round')
                season_counts[season] = season_counts.get(season, 0) + 1

            total_season_mentions = sum(season_counts.values())

            if total_season_mentions > 0:
                # Primary season focus
                primary_season = max(season_counts.items(), key=lambda x: x[1])[0]
                features['primary_season_focus'] = primary_season
                features['seasonal_concentration'] = season_counts[primary_season] / total_season_mentions

                # Season mix indicators
                features['winter_equipment_ratio'] = season_counts['winter'] / total_season_mentions
                features['summer_equipment_ratio'] = season_counts['summer'] / total_season_mentions
                features['year_round_equipment_ratio'] = season_counts['year_round'] / total_season_mentions

                # Season diversity (0-1, higher = more diverse)
                season_proportions = [v / total_season_mentions for v in season_counts.values() if v > 0]
                if len(season_proportions) > 1:
                    # Calculate normalized entropy
                    entropy = -sum(p * np.log(p) for p in season_proportions)
                    max_entropy = np.log(len(season_proportions))
                    features['seasonal_equipment_mix'] = entropy / max_entropy
                else:
                    features['seasonal_equipment_mix'] = 0  # Only one season

                # Binary indicators
                features['has_winter_equipment'] = 1 if season_counts['winter'] > 0 else 0
                features['has_summer_equipment'] = 1 if season_counts['summer'] > 0 else 0
                features['has_multi_season'] = 1 if len([s for s, c in season_counts.items() if c > 0]) > 1 else 0
            else:
                features['primary_season_focus'] = 'none'
                features['seasonal_concentration'] = 0
                features['winter_equipment_ratio'] = 0
                features['summer_equipment_ratio'] = 0
                features['year_round_equipment_ratio'] = 0
                features['seasonal_equipment_mix'] = 0
                features['has_winter_equipment'] = 0
                features['has_summer_equipment'] = 0
                features['has_multi_season'] = 0

            # ========== FEATURE 4: EQUIPMENT MATURITY LEVEL ==========
            # Based on equipment choices, assign maturity level
            maturity_score = 0
            maturity_components = []

            # Component 1: Equipment complexity
            avg_complexity = features.get('final_complexity', 1.5)
            maturity_components.append(min(avg_complexity / 4, 1))  # Normalize to 0-1

            # Component 2: Upgrade trajectory
            if features.get('has_upgrade', 0) == 1:
                maturity_components.append(0.8)  # Upgrading indicates learning/improvement
            elif features.get('has_downgrade', 0) == 1:
                maturity_components.append(0.2)  # Downgrading might indicate budget constraints
            else:
                maturity_components.append(0.5)  # Neutral

            # Component 3: Season diversity
            if features.get('has_multi_season', 0) == 1:
                maturity_components.append(0.7)  # Considering multiple seasons
            else:
                maturity_components.append(0.3)  # Single season focus

            # Component 4: Equipment variety (moderate is good)
            variety_index = features.get('equipment_variety_index', 0)
            if 0.3 <= variety_index <= 0.7:
                maturity_components.append(0.6)  # Balanced exploration
            elif variety_index < 0.3:
                maturity_components.append(0.4)  # Too focused
            else:
                maturity_components.append(0.3)  # Too scattered

            features['equipment_maturity_level'] = np.mean(maturity_components) if maturity_components else 0.5

            # ========== FEATURE 5: EQUIPMENT STRATEGY SIGNALS ==========
            # What does their equipment pattern suggest about their strategy?

            # Signal 1: Replacement vs Upgrade
            if features['equipment_family_consistency'] == 1 and features['has_upgrade'] == 0:
                features['likely_replacement'] = 1  # Same type, no upgrade = replacement
            else:
                features['likely_replacement'] = 0

            # Signal 2: System expansion
            if features['equipment_variety_count'] > 1 and features['has_multi_season'] == 1:
                features['likely_system_expansion'] = 1  # Different equipment for different needs
            else:
                features['likely_system_expansion'] = 0

            # Signal 3: Technology adoption
            high_complexity_equipment = ['HEAT_PUMP', 'GEOTHERMAL', 'SOLAR_THERMAL', 'HYBRID_SYSTEM']
            has_high_tech = any(equip in high_complexity_equipment for equip in equipment_sequence)
            features['early_technology_adopter'] = 1 if has_high_tech else 0

        upgrade_features.append(features)

    # Convert to DataFrame
    upgrade_features_df = pd.DataFrame(upgrade_features)

    # Report statistics
    print(f"\n✅ Created {len(upgrade_features_df.columns) - 1} equipment upgrade features")
    print(f"   Samples: {len(upgrade_features_df):,}")

    # Show key feature distributions
    if len(upgrade_features_df) > 0:
        key_features = [
            'equipment_family_consistency',
            'upgrade_trajectory_score',
            'has_upgrade',
            'seasonal_equipment_mix',
            'equipment_maturity_level',
            'equipment_variety_index',
            'has_multi_season',
            'early_technology_adopter'
        ]

        print("\n📊 KEY UPGRADE FEATURES SUMMARY:")
        print("-" * 60)
        for feat in key_features:
            if feat in upgrade_features_df.columns:
                mean_val = upgrade_features_df[feat].mean()
                std_val = upgrade_features_df[feat].std()
                non_zero = (upgrade_features_df[feat] != 0).mean() * 100
                print(f"{feat:35} : mean={mean_val:.3f}, std={std_val:.3f}, non-zero={non_zero:.1f}%")

    return upgrade_features_df