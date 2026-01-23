import pandas as pd
import numpy as np
import re


def create_model_features(df):
    """
    PILLAR 1.2: Model Complexity & Specialization Features

    Creates features from 'modele_produit' column to capture:
    1. Model name complexity and structure
    2. Technical vs standard model indicators
    3. Product series consistency
    4. Model sophistication signals
    """
    print("=" * 80)
    print("CREATING MODEL COMPLEXITY & SPECIALIZATION FEATURES")
    print("=" * 80)

    # Check if model data is available
    if 'modele_produit' not in df.columns:
        print("⚠️ WARNING: 'modele_produit' column not found in dataset")
        print("  Returning empty model features DataFrame")
        return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing model data for {df['numero_compte'].nunique():,} customers")
    print(f"Total quotes with model info: {df['modele_produit'].notna().sum():,}")
    print(f"Unique models in dataset: {df['modele_produit'].nunique():,}")

    # Sort by customer and date
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    model_features = []

    # Define model sophistication indicators
    technical_indicators = [
        'PRO', 'EXPERT', 'TECH', 'ADVANCED', 'PREMIUM', 'ELITE', 'PERFORMANCE',
        'PLATINUM', 'GOLD', 'SILVER', 'CONDENS', 'INVERTER', 'VRF', 'COP'
    ]

    standard_indicators = [
        'BASIC', 'STANDARD', 'CLASSIC', 'ESSENTIAL', 'SIMPLE', 'ENTRY', 'START'
    ]

    # Energy efficiency classes
    efficiency_classes = ['A+++', 'A++', 'A+', 'A', 'B', 'C', 'D']

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        model_data = customer_data['modele_produit'].dropna().astype(str)

        if len(model_data) == 0:
            # No model data for this customer
            features.update({
                'model_data_available': 0,
                'avg_model_name_length': 0,
                'model_name_complexity': 0,
                'technical_model_ratio': 0,
                'standard_model_ratio': 0,
                'model_variety_score': 0,
                'series_consistency': 0,
                'has_efficiency_class': 0,
                'model_sophistication_score': 0
            })
        else:
            features['model_data_available'] = 1

            # ========== FEATURE 1: MODEL NAME LENGTH & COMPLEXITY ==========
            model_lengths = model_data.str.len()
            features['avg_model_name_length'] = model_lengths.mean()

            # Complexity: standard deviation of lengths (variation indicates complexity)
            features['model_name_complexity'] = model_lengths.std()

            # ========== FEATURE 2: TECHNICAL VS STANDARD INDICATORS ==========
            # Check for technical specifications in model names
            technical_count = 0
            standard_count = 0
            efficiency_found = 0

            for model_name in model_data:
                model_upper = model_name.upper()

                # Technical indicators
                if any(indicator in model_upper for indicator in technical_indicators):
                    technical_count += 1

                # Standard/basic indicators
                if any(indicator in model_upper for indicator in standard_indicators):
                    standard_count += 1

                # Energy efficiency classes
                if any(eclass in model_upper for eclass in efficiency_classes):
                    efficiency_found += 1

            features['technical_model_ratio'] = technical_count / len(model_data)
            features['standard_model_ratio'] = standard_count / len(model_data)
            features['has_efficiency_class'] = 1 if efficiency_found > 0 else 0

            # ========== FEATURE 3: MODEL VARIETY SCORE ==========
            unique_models = model_data.nunique()
            features['model_variety_score'] = unique_models / len(model_data)

            # ========== FEATURE 4: SERIES CONSISTENCY ==========
            # Extract potential series names (first word or code before space/number)
            def extract_series(model_name):
                # Remove special characters, split
                cleaned = re.sub(r'[^\w\s]', ' ', model_name)
                parts = cleaned.split()

                if len(parts) > 0:
                    # Look for alphanumeric codes that might be series names
                    for part in parts:
                        if (len(part) >= 2 and
                                any(c.isalpha() for c in part) and
                                any(c.isdigit() for c in part)):
                            return part
                    return parts[0]
                return model_name[:10]  # First 10 chars if no clear split

            series_names = model_data.apply(extract_series)
            unique_series = series_names.nunique()
            features['series_consistency'] = 1 if unique_series == 1 else 0
            features['unique_series_count'] = unique_series

            # ========== FEATURE 5: MODEL SOPHISTICATION SCORE ==========
            # Composite score of model complexity
            sophistication_components = []

            # Longer names often indicate more specifications
            if features['avg_model_name_length'] > 0:
                norm_length = min(features['avg_model_name_length'] / 50, 1)  # Cap at 50 chars
                sophistication_components.append(norm_length)

            # Technical indicators
            sophistication_components.append(features['technical_model_ratio'])

            # Efficiency class presence
            sophistication_components.append(features['has_efficiency_class'])

            # Low standard model ratio (inverse relationship)
            sophistication_components.append(1 - features['standard_model_ratio'])

            features['model_sophistication_score'] = np.mean(sophistication_components)

            # ========== FEATURE 6: NUMERIC SPECIFICATIONS ==========
            # Extract kW, BTU, SEER ratings etc.
            kw_values = []
            for model_name in model_data:
                # Look for kW specifications (e.g., "25KW", "10.5KW")
                kw_match = re.search(r'(\d+(?:\.\d+)?)\s*KW', model_name, re.IGNORECASE)
                if kw_match:
                    kw_values.append(float(kw_match.group(1)))

            if kw_values:
                features['avg_kw_rating'] = np.mean(kw_values)
                features['kw_range'] = max(kw_values) - min(kw_values) if len(kw_values) > 1 else 0
            else:
                features['avg_kw_rating'] = 0
                features['kw_range'] = 0

            # ========== FEATURE 7: MODEL TYPE CONSISTENCY ==========
            # Categorize models by type keywords
            model_types = {
                'condensing': ['CONDENS', 'CONDENSE', 'COND'],
                'inverter': ['INVERTER', 'INV', 'VRF'],
                'heat_pump': ['PAC', 'POMPE', 'HEAT PUMP'],
                'boiler': ['CHAUFFE', 'BOILER', 'CHAUDIERE'],
                'stove': ['POELE', 'STOVE', 'CHEMINEE']
            }

            type_counts = {t: 0 for t in model_types.keys()}
            for model_name in model_data:
                model_upper = model_name.upper()
                for type_name, keywords in model_types.items():
                    if any(keyword in model_upper for keyword in keywords):
                        type_counts[type_name] += 1

            # Dominant type
            if sum(type_counts.values()) > 0:
                dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
                features['dominant_model_type'] = dominant_type
                features['model_type_concentration'] = max(type_counts.values()) / len(model_data)
            else:
                features['dominant_model_type'] = 'unknown'
                features['model_type_concentration'] = 0

        model_features.append(features)

    # Convert to DataFrame
    model_features_df = pd.DataFrame(model_features)

    # Report statistics
    print(f"\n✅ Created {len(model_features_df.columns) - 1} model complexity features")
    print(f"   Samples: {len(model_features_df):,}")

    # Show feature distribution
    if len(model_features_df) > 0:
        numeric_features = [col for col in model_features_df.columns
                            if col != 'numero_compte' and
                            model_features_df[col].dtype in ['int64', 'float64']]

        print("\n📊 MODEL COMPLEXITY FEATURES SUMMARY:")
        print("-" * 50)
        for feat in numeric_features[:10]:
            mean_val = model_features_df[feat].mean()
            std_val = model_features_df[feat].std()
            print(f"{feat:30} : mean={mean_val:.3f}, std={std_val:.3f}")

    return model_features_df