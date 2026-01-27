import pandas as pd
import numpy as np


def create_equipment_features(df):
    """
    VECTORIZED VERSION of equipment upgrade features with chronological sorting
    """
    print("=" * 80)
    print("CREATING EQUIPMENT UPGRADE PATH FEATURES (VECTORIZED)")
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

    # ========== CRITICAL: FORCE CHRONOLOGICAL SORTING ==========
    if 'dt_creation_devis' in df.columns:
        print("🔍 Ensuring chronological order to prevent leakage...")
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)
        print("✅ Data sorted chronologically by customer and date")
    else:
        print("⚠️  No date column - using existing order (potential leakage risk)")
        df = df.sort_values(['numero_compte']).reset_index(drop=True)

    # ========== DEFINE MAPPINGS ==========
    equipment_complexity = {
        'STOVE': 1, 'Poêle': 1, 'CHEMINEE': 1, 'INSERT': 1,
        'BOILER_GAS': 2, 'Chaudière': 2, 'BOILER_OIL': 2,
        'AIR_CONDITIONER': 2, 'Climatisation': 2,
        'Plomberie Sanitaire': 2, 'SANITARY': 2,
        'HEAT_PUMP': 3, 'Pompe à chaleur': 3,
        'CONDENSING_BOILER': 3, 'CHAUDIERE_CONDENSATION': 3,
        'VRF_SYSTEM': 3, 'SYSTEME_VRF': 3,
        'GEOTHERMAL': 4, 'GEOTHERMIE': 4,
        'SOLAR_THERMAL': 4, 'SOLAIRE_THERMIQUE': 4,
        'HYBRID_SYSTEM': 4, 'SYSTEME_HYBRIDE': 4,
        'OTHER': 1.5, 'AUTRES': 1.5
    }

    equipment_seasonality = {
        'STOVE': 'winter', 'Poêle': 'winter',
        'BOILER_GAS': 'winter', 'Chaudière': 'winter',
        'BOILER_OIL': 'winter', 'HEAT_PUMP': 'winter',
        'CONDENSING_BOILER': 'winter', 'GEOTHERMAL': 'winter',
        'AIR_CONDITIONER': 'summer', 'Climatisation': 'summer',
        'VRF_SYSTEM': 'summer',
        'Plomberie Sanitaire': 'year_round', 'SANITARY': 'year_round',
        'SOLAR_THERMAL': 'year_round', 'HYBRID_SYSTEM': 'year_round',
        'OTHER': 'year_round', 'AUTRES': 'year_round'
    }

    upgrade_paths = {
        ('STOVE', 'BOILER_GAS'): 'basic_to_primary_heating',
        ('STOVE', 'HEAT_PUMP'): 'basic_to_advanced',
        ('BOILER_GAS', 'HEAT_PUMP'): 'primary_to_advanced',
        ('BOILER_GAS', 'CONDENSING_BOILER'): 'standard_to_efficient',
        ('AIR_CONDITIONER', 'HEAT_PUMP'): 'cooling_to_hybrid',
        ('BOILER_GAS', 'GEOTHERMAL'): 'conventional_to_renewable'
    }

    high_complexity_equipment = {'HEAT_PUMP', 'GEOTHERMAL', 'SOLAR_THERMAL', 'HYBRID_SYSTEM'}

    # ========== VECTORIZED COMPLEXITY & SEASONALITY MAPPING ==========
    print("\n📊 Applying complexity and seasonality mappings...")

    # Create a copy with only needed columns
    df_work = df[['numero_compte', equipment_col]].copy()

    # Map equipment to complexity and seasonality
    df_work['equipment_complexity'] = df_work[equipment_col].map(equipment_complexity).fillna(1.5)
    df_work['equipment_season'] = df_work[equipment_col].map(equipment_seasonality).fillna('year_round')
    df_work['is_high_tech'] = df_work[equipment_col].isin(high_complexity_equipment).astype(int)

    # Filter out customers with no equipment data
    valid_customers = df_work[df_work[equipment_col].notna()]['numero_compte'].unique()
    print(f"Customers with equipment data: {len(valid_customers):,}")

    # ========== GROUP-BY AGGREGATIONS (VECTORIZED) ==========
    print("\n🔧 Computing group-by aggregations...")

    # Group 1: Basic equipment stats
    group1 = df_work.groupby('numero_compte').agg(
        equipment_upgrade_data_available=('equipment_complexity', lambda x: 1 if len(x) > 0 else 0),
        equipment_variety_count=(equipment_col, 'nunique'),
        final_complexity=('equipment_complexity', 'last'),
        first_equipment=(equipment_col, 'first'),
        last_equipment=(equipment_col, 'last'),
        quote_count=(equipment_col, 'size'),
        has_high_tech=('is_high_tech', 'max')
    ).reset_index()

    # Group 2: Complexity statistics
    complexity_stats = df_work.groupby('numero_compte')['equipment_complexity'].agg([
        ('min_complexity', 'min'),
        ('max_complexity', 'max'),
        ('mean_complexity', 'mean'),
        ('std_complexity', 'std')
    ]).reset_index()

    complexity_stats['complexity_range'] = complexity_stats['max_complexity'] - complexity_stats['min_complexity']

    # Group 3: Seasonality statistics
    # Create dummy columns for each season
    season_dummies = pd.get_dummies(df_work['equipment_season'], prefix='season')
    df_season = pd.concat([df_work[['numero_compte']], season_dummies], axis=1)

    season_counts = df_season.groupby('numero_compte').sum().reset_index()

    # Calculate season ratios
    total_seasons = season_counts[['season_winter', 'season_summer', 'season_year_round']].sum(axis=1)

    for season in ['winter', 'summer', 'year_round']:
        col = f'season_{season}'
        if col in season_counts.columns:
            season_counts[f'{season}_equipment_ratio'] = season_counts[col] / total_seasons.replace(0, 1)

    # Primary season focus
    season_cols = [c for c in season_counts.columns if c.startswith('season_') and c != 'season_nan']
    season_counts['primary_season_focus'] = season_counts[season_cols].idxmax(axis=1).str.replace('season_', '')

    # ========== MERGE ALL GROUP STATS ==========
    print("🔄 Merging aggregated features...")

    result = pd.merge(group1, complexity_stats, on='numero_compte', how='left')
    result = pd.merge(result, season_counts, on='numero_compte', how='left')

    # ========== VECTORIZED FEATURE CALCULATIONS ==========
    print("⚡ Computing derived features...")

    # Feature 1: Equipment Family Consistency
    result['equipment_family_consistency'] = (result['equipment_variety_count'] == 1).astype(int)

    # Feature 2: Equipment Variety Index (using vectorized operations)
    max_possible_variety = result[['equipment_variety_count', 'quote_count']].min(axis=1)
    result['equipment_variety_index'] = np.where(
        max_possible_variety > 1,
        (result['equipment_variety_count'] - 1) / (max_possible_variety - 1),
        0
    )

    # Feature 3: Upgrade Trajectory Score (vectorized linear regression per group)
    print("📈 Calculating upgrade trajectories...")

    def calculate_slope(group):
        if len(group) < 2:
            return 0
        x = np.arange(len(group))
        if len(group) == 0 or np.all(group == group.iloc[0]):  # <-- FIXED
            return 0
        return np.polyfit(x, group, 1)[0]

    # Calculate slopes for each customer
    slopes = df_work.groupby('numero_compte')['equipment_complexity'].apply(calculate_slope)
    slopes.name = 'upgrade_trajectory_score'

    result = result.merge(slopes, on='numero_compte', how='left')
    result['upgrade_trajectory_score'] = result['upgrade_trajectory_score'].fillna(0)

    # Feature 4: Upgrade indicators
    result['has_upgrade'] = ((result['last_equipment'].map(equipment_complexity).fillna(1.5) >
                              result['first_equipment'].map(equipment_complexity).fillna(1.5)).astype(int))
    result['has_downgrade'] = ((result['last_equipment'].map(equipment_complexity).fillna(1.5) <
                                result['first_equipment'].map(equipment_complexity).fillna(1.5)).astype(int))

    # Check specific upgrade paths
    def get_upgrade_type(first, last):
        key = (first, last)
        return upgrade_paths.get(key, 'none')

    result['specific_upgrade_type'] = result.apply(
        lambda row: get_upgrade_type(row['first_equipment'], row['last_equipment']),
        axis=1
    )
    result['has_upgrade_pattern'] = (result['specific_upgrade_type'] != 'none').astype(int)

    # Feature 5: Equipment complexity trend
    result['equipment_complexity_trend'] = np.select(
        [
            result['upgrade_trajectory_score'] > 0.05,
            result['upgrade_trajectory_score'] < -0.05
        ],
        [1, -1],
        default=0
    )

    # Feature 6: Seasonal mix (entropy-based)
    def calculate_season_entropy(row):
        seasons = ['winter', 'summer', 'year_round']
        counts = [row.get(f'season_{s}', 0) for s in seasons]
        total = sum(counts)

        if total == 0:
            return 0

        proportions = [c / total for c in counts if c > 0]
        if len(proportions) <= 1:
            return 0

        entropy = -sum(p * np.log(p) for p in proportions)
        max_entropy = np.log(len(proportions))
        return entropy / max_entropy

    result['seasonal_equipment_mix'] = result.apply(calculate_season_entropy, axis=1)

    # Feature 7: Binary season indicators
    for season in ['winter', 'summer']:
        result[f'has_{season}_equipment'] = (result[f'season_{season}'] > 0).astype(int)

    result['has_multi_season'] = (
                (result[['season_winter', 'season_summer', 'season_year_round']] > 0).sum(axis=1) > 1).astype(int)

    # Feature 8: Equipment maturity level (vectorized calculation)
    maturity_components = []

    # Component 1: Equipment complexity (normalized to 0-1)
    maturity_components.append(np.minimum(result['final_complexity'] / 4, 1))

    # Component 2: Upgrade trajectory
    maturity_components.append(np.select(
        [result['has_upgrade'] == 1, result['has_downgrade'] == 1],
        [0.8, 0.2],
        default=0.5
    ))

    # Component 3: Season diversity
    maturity_components.append(np.select(
        [result['has_multi_season'] == 1],
        [0.7],
        default=0.3
    ))

    # Component 4: Equipment variety (balanced is best)
    maturity_components.append(np.select(
        [
            (result['equipment_variety_index'] >= 0.3) & (result['equipment_variety_index'] <= 0.7),
            result['equipment_variety_index'] < 0.3
        ],
        [0.6, 0.4],
        default=0.3
    ))

    # Calculate average maturity
    maturity_matrix = np.column_stack(maturity_components)
    result['equipment_maturity_level'] = np.mean(maturity_matrix, axis=1)

    # Feature 9: Strategy signals
    result['likely_replacement'] = ((result['equipment_family_consistency'] == 1) &
                                    (result['has_upgrade'] == 0)).astype(int)

    result['likely_system_expansion'] = ((result['equipment_variety_count'] > 1) &
                                         (result['has_multi_season'] == 1)).astype(int)

    result['early_technology_adopter'] = result['has_high_tech']

    # Feature 10: Seasonal concentration
    result['seasonal_concentration'] = result[['season_winter', 'season_summer', 'season_year_round']].max(axis=1) / \
                                       result[['season_winter', 'season_summer', 'season_year_round']].sum(
                                           axis=1).replace(0, 1)

    # ========== FILL MISSING VALUES ==========
    print("🧹 Cleaning up missing values...")

    # Fill NaN values for customers without equipment data
    equipment_cols = [col for col in result.columns if col != 'numero_compte']
    result[equipment_cols] = result[equipment_cols].fillna(0)

    # For categorical columns
    categorical_fills = {
        'primary_season_focus': 'none',
        'specific_upgrade_type': 'none'
    }

    for col, fill_value in categorical_fills.items():
        if col in result.columns:
            result[col] = result[col].fillna(fill_value)

    # ========== FINAL REPORT ==========
    print(f"\n✅ Created {len(result.columns) - 1} equipment upgrade features")
    print(f"   Samples: {len(result):,} customers")

    # Show key feature distributions
    if len(result) > 0:
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
            if feat in result.columns:
                mean_val = result[feat].mean()
                std_val = result[feat].std()
                non_zero = (result[feat] != 0).mean() * 100
                print(f"{feat:35} : mean={mean_val:.3f}, std={std_val:.3f}, non-zero={non_zero:.1f}%")

    return result