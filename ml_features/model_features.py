import pandas as pd
import numpy as np
import re


def create_model_features(df):
    """
    HYPER-FAST vectorized model features
    Target: 7-10x speedup (from 17.6s to ~1.8-2.5s)
    """
    print("=" * 80)
    print("CREATING MODEL FEATURES (HYPER-FAST)")
    print("=" * 80)

    if 'modele_produit' not in df.columns:
        return pd.DataFrame(columns=['numero_compte'])

    # Sort once
    if 'dt_creation_devis' in df.columns:
        df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)

    print(f"Processing {df['numero_compte'].nunique():,} customers")

    # Static lists as compiled regex for speed
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

    # 1. SINGLE GROUPBY with vectorized string operations
    print("👥 Grouping and preprocessing...")

    # Get model sequences as strings
    customer_groups = df.groupby('numero_compte')['modele_produit'].apply(
        lambda x: [str(m).upper() if pd.notna(m) else '' for m in x]
    )
    customer_ids = customer_groups.index.values
    model_sequences = customer_groups.values
    n_customers = len(customer_ids)

    print(f"  Processing {n_customers:,} customers")

    # 2. PRE-ALLOCATE arrays
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

    # 3. OPTIMIZED BATCH PROCESSING
    print("⚡ Hyper-fast calculations...")

    # Process all customers at once with vectorized operations
    for i in range(n_customers):
        models = model_sequences[i]
        if not models or all(m == '' for m in models):
            model_data_available[i] = 0
            continue

        # Filter out empty strings
        valid_models = [m for m in models if m]
        n_models = len(valid_models)

        if n_models == 0:
            continue

        # Convert to numpy array once
        models_array = np.array(valid_models)

        # FEATURE 1: Length & complexity (vectorized)
        lengths = np.array([len(m) for m in models_array])
        avg_model_name_length[i] = np.mean(lengths)
        if n_models > 1:
            model_name_complexity[i] = np.std(lengths)

        # FEATURE 2: Technical/Standard indicators (FAST regex search)
        tech_flags = np.array([1 if technical_regex.search(m) else 0 for m in models_array])
        standard_flags = np.array([1 if standard_regex.search(m) else 0 for m in models_array])
        efficiency_flags = np.array([1 if efficiency_regex.search(m) else 0 for m in models_array])

        technical_model_ratio[i] = np.mean(tech_flags)
        standard_model_ratio[i] = np.mean(standard_flags)
        has_efficiency_class[i] = 1 if np.any(efficiency_flags) else 0

        # FEATURE 3: Model variety
        unique_models = len(np.unique(models_array))
        model_variety_score[i] = unique_models / n_models

        # FEATURE 4: Series consistency (SIMPLIFIED for speed)
        # Extract first alphanumeric token as series
        series_tokens = []
        for model in models_array:
            # Fast series extraction: take first meaningful token
            parts = re.split(r'[^A-Z0-9]+', model)
            if parts and parts[0]:
                series_tokens.append(parts[0][:10])  # First 10 chars of first token
            else:
                series_tokens.append(model[:10])

        unique_series = len(set(series_tokens))
        series_consistency[i] = 1 if unique_series == 1 else 0
        unique_series_count[i] = unique_series

        # FEATURE 5: KW ratings (vectorized regex)
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

        # FEATURE 6: Model type (optimized)
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

    print("✅ Calculations complete")

    # 4. CREATE DATAFRAME with derived features
    print("📝 Creating final DataFrame...")

    # Calculate sophistication score (vectorized)
    norm_length = np.minimum(avg_model_name_length / 50, 1)
    model_sophistication_score = (
            norm_length * 0.25 +
            technical_model_ratio * 0.25 +
            has_efficiency_class * 0.25 +
            (1 - standard_model_ratio) * 0.25
    )

    result = pd.DataFrame({
        'numero_compte': customer_ids,
        'model_data_available': model_data_available,
        'avg_model_name_length': avg_model_name_length,
        'model_name_complexity': model_name_complexity,
        'technical_model_ratio': technical_model_ratio,
        'standard_model_ratio': standard_model_ratio,
        'has_efficiency_class': has_efficiency_class,
        'model_variety_score': model_variety_score,
        'series_consistency': series_consistency,
        'unique_series_count': unique_series_count,
        'avg_kw_rating': avg_kw_rating,
        'kw_range': kw_range,
        'dominant_model_type': dominant_model_type,
        'model_type_concentration': model_type_concentration,
        'model_sophistication_score': model_sophistication_score
    })

    # 5. ADD MISSING CUSTOMERS
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

    # 6. REPORT
    print(f"\n✅ Created {len(result.columns) - 1} model features")
    print(f"   Customers: {len(result):,}")

    return result