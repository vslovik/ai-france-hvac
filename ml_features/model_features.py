import pandas as pd
import numpy as np
import re


def create_model_features(df):
    """
    ENHANCED HYPER-FAST vectorized model features
    Now includes new product columns and model sophistication
    """
    print("=" * 80)
    print("CREATING ENHANCED MODEL FEATURES (HYPER-FAST)")
    print("=" * 80)

    # ========== 1. BASE MODEL FEATURES ==========

    if 'modele_produit' not in df.columns:
        return pd.DataFrame(columns=['numero_compte'])

    # Sort once for sequence features
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)

    print(f"Processing {df['numero_compte'].nunique():,} customers")

    # Compile regex patterns once
    technical_regex = re.compile(
        r'PRO|EXPERT|TECH|ADVANCED|PREMIUM|ELITE|PERFORMANCE|PLATINUM|GOLD|SILVER|CONDENS|INVERTER|VRF|COP',
        re.IGNORECASE)
    standard_regex = re.compile(r'BASIC|STANDARD|CLASSIC|ESSENTIAL|SIMPLE|ENTRY|START', re.IGNORECASE)
    efficiency_regex = re.compile(r'A\+\+\+|A\+\+|A\+|A|B|C|D', re.IGNORECASE)
    kw_regex = re.compile(r'(\d+(?:\.\d+)?)\s*KW', re.IGNORECASE)

    # Model type regexes
    type_regexes = {
        'condensing': re.compile(r'CONDENS|CONDENSE|COND', re.IGNORECASE),
        'inverter': re.compile(r'INVERTER|INV|VRF', re.IGNORECASE),
        'heat_pump': re.compile(r'PAC|POMPE|HEAT PUMP', re.IGNORECASE),
        'boiler': re.compile(r'CHAUFFE|BOILER|CHAUDIERE', re.IGNORECASE),
        'stove': re.compile(r'POELE|STOVE|CHEMINEE', re.IGNORECASE)
    }

    # ========== 2. GET MODEL SEQUENCES ==========

    customer_groups = df.groupby('numero_compte')['modele_produit'].apply(
        lambda x: [str(m).upper() if pd.notna(m) else '' for m in x]
    )
    customer_ids = customer_groups.index.values
    model_sequences = customer_groups.values
    n_customers = len(customer_ids)

    # ========== 3. PRE-ALLOCATE ARRAYS ==========

    model_data_available = np.ones(n_customers, dtype=int)
    avg_model_name_length = np.zeros(n_customers)
    model_name_complexity = np.zeros(n_customers)
    technical_model_ratio = np.zeros(n_customers)
    standard_model_ratio = np.zeros(n_customers)
    has_efficiency_class = np.zeros(n_customers, dtype=int)
    model_variety_score = np.zeros(n_customers)
    series_consistency = np.zeros(n_customers, dtype=int)
    unique_series_count = np.zeros(n_customers, dtype=int)
    avg_kw_rating = np.zeros(n_customers)
    kw_range = np.zeros(n_customers)
    model_type_concentration = np.zeros(n_customers)
    dominant_model_type = np.full(n_customers, 'unknown', dtype=object)
    model_name_std = np.zeros(n_customers)  # NEW: Standard deviation of model name length
    efficiency_class_score = np.zeros(n_customers)  # NEW: Numeric efficiency score

    # ========== 4. VECTORIZED BATCH PROCESSING ==========
    print("⚡ Hyper-fast calculations...")

    for i in range(n_customers):
        models = model_sequences[i]
        if not models or all(m == '' for m in models):
            model_data_available[i] = 0
            continue

        valid_models = [m for m in models if m]
        n_models = len(valid_models)

        if n_models == 0:
            continue

        models_array = np.array(valid_models)

        # Feature 1: Length & complexity
        lengths = np.array([len(m) for m in models_array])
        avg_model_name_length[i] = np.mean(lengths)
        model_name_std[i] = np.std(lengths) if n_models > 1 else 0

        # Feature 2: Technical/Standard indicators
        tech_flags = np.array([1 if technical_regex.search(m) else 0 for m in models_array])
        standard_flags = np.array([1 if standard_regex.search(m) else 0 for m in models_array])
        efficiency_flags = np.array([1 if efficiency_regex.search(m) else 0 for m in models_array])

        technical_model_ratio[i] = np.mean(tech_flags)
        standard_model_ratio[i] = np.mean(standard_flags)
        has_efficiency_class[i] = 1 if np.any(efficiency_flags) else 0

        # Feature 3: Efficiency class score (A+++ = 5, A = 4, etc.)
        efficiency_scores = {'A+++': 5, 'A++': 4, 'A+': 3, 'A': 2, 'B': 1, 'C': 0, 'D': -1}
        eff_values = []
        for m in models_array:
            match = efficiency_regex.search(m)
            if match:
                eff_class = match.group(0).upper()
                eff_values.append(efficiency_scores.get(eff_class, 0))
        if eff_values:
            efficiency_class_score[i] = np.mean(eff_values)

        # Feature 4: Model variety
        unique_models = len(np.unique(models_array))
        model_variety_score[i] = unique_models / n_models

        # Feature 5: Series consistency
        series_tokens = []
        for model in models_array:
            parts = re.split(r'[^A-Z0-9]+', model)
            if parts and parts[0]:
                series_tokens.append(parts[0][:10])
            else:
                series_tokens.append(model[:10])

        unique_series = len(set(series_tokens))
        series_consistency[i] = 1 if unique_series == 1 else 0
        unique_series_count[i] = unique_series

        # Feature 6: KW ratings
        kw_values = []
        for model in models_array:
            match = kw_regex.search(model)
            if match:
                kw_values.append(float(match.group(1)))

        if kw_values:
            kw_array = np.array(kw_values)
            avg_kw_rating[i] = np.mean(kw_array)
            if len(kw_values) > 1:
                kw_range[i] = np.max(kw_array) - np.min(kw_array)

        # Feature 7: Model type concentration
        type_counts = np.zeros(len(type_regexes))
        type_names = list(type_regexes.keys())

        for model in models_array:
            for idx, (_, regex) in enumerate(type_regexes.items()):
                if regex.search(model):
                    type_counts[idx] += 1

        total_matches = np.sum(type_counts)
        if total_matches > 0:
            dominant_idx = np.argmax(type_counts)
            dominant_model_type[i] = type_names[dominant_idx]
            model_type_concentration[i] = np.max(type_counts) / n_models

    # ========== 5. CREATE DATAFRAME ==========
    print("📝 Creating final DataFrame...")

    # Model sophistication score (enhanced)
    norm_length = np.minimum(avg_model_name_length / 50, 1)
    norm_std = np.minimum(model_name_std / 20, 1)

    model_sophistication_score = (
            norm_length * 0.20 +
            norm_std * 0.15 +
            technical_model_ratio * 0.20 +
            has_efficiency_class * 0.15 +
            (1 - standard_model_ratio) * 0.15 +
            efficiency_class_score / 10 * 0.15
    )

    result = pd.DataFrame({
        'numero_compte': customer_ids,
        'model_data_available': model_data_available,
        'avg_model_name_length': avg_model_name_length,
        'model_name_complexity': model_name_std,
        'technical_model_ratio': technical_model_ratio,
        'standard_model_ratio': standard_model_ratio,
        'has_efficiency_class': has_efficiency_class,
        'efficiency_class_score': efficiency_class_score,
        'model_variety_score': model_variety_score,
        'series_consistency': series_consistency,
        'unique_series_count': unique_series_count,
        'avg_kw_rating': avg_kw_rating,
        'kw_range': kw_range,
        'dominant_model_type': dominant_model_type,
        'model_type_concentration': model_type_concentration,
        'model_sophistication_score': model_sophistication_score
    })

    # ========== 6. ADD MISSING CUSTOMERS ==========
    all_customers = df['numero_compte'].unique()
    if len(result) < len(all_customers):
        missing_mask = ~pd.Series(all_customers).isin(customer_ids)
        missing_customers = all_customers[missing_mask]

        if len(missing_customers) > 0:
            missing_df = pd.DataFrame({
                'numero_compte': missing_customers,
                'model_data_available': 0,
                'avg_model_name_length': 0,
                'model_name_complexity': 0,
                'technical_model_ratio': 0,
                'standard_model_ratio': 0,
                'has_efficiency_class': 0,
                'efficiency_class_score': 0,
                'model_variety_score': 0,
                'series_consistency': 0,
                'unique_series_count': 0,
                'avg_kw_rating': 0,
                'kw_range': 0,
                'dominant_model_type': 'unknown',
                'model_type_concentration': 0,
                'model_sophistication_score': 0
            })
            result = pd.concat([result, missing_df], ignore_index=True)

    print(f"\n✅ Created {len(result.columns) - 1} model features")
    print(f"   Customers: {len(result):,}")
    print(f"   NEW: model_name_complexity (std), efficiency_class_score")

    return result