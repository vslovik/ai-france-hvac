import numpy as np
import pandas as pd


def create_engagement_features(df_quotes):
    """
    Create engagement features from raw quote data
    """
    # Sort by customer and date
    df = df_quotes.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    # Group by customer
    customer_stats = df.groupby('numero_compte').agg(
        total_quotes=('id_devis', 'count'),
        first_quote_date=('dt_creation_devis', 'min'),
        last_quote_date=('dt_creation_devis', 'max'),
        unique_agencies=('nom_agence', 'nunique'),
        unique_equipment=('regroup_famille_equipement_produit', 'nunique'),
        unique_brands=('marque_produit', 'nunique')
    ).reset_index()

    # Calculate time-based features
    customer_stats['engagement_days'] = (
            pd.to_datetime(customer_stats['last_quote_date']) -
            pd.to_datetime(customer_stats['first_quote_date'])
    ).dt.days

    customer_stats['engagement_density'] = customer_stats['total_quotes'] / (
            customer_stats['engagement_days'] + 1  # +1 to avoid division by zero
    )

    # Single vs multi quote
    customer_stats['is_single_quote'] = (customer_stats['total_quotes'] == 1).astype(int)

    return customer_stats[['numero_compte', 'total_quotes', 'engagement_days',
                           'engagement_density', 'is_single_quote',
                           'unique_equipment', 'unique_brands']]


def create_equipment_brand_features(df_quotes):
    """
    Create equipment and brand preference features
    """
    # Equipment complexity mapping
    equipment_complexity = {
        'STOVE': 1,
        'BOILER_GAS': 2,
        'AIR_CONDITIONER': 2,
        'HEAT_PUMP': 3,
        'GEOTHERMAL': 4,
        'SOLAR_THERMAL': 4
    }

    # Get most frequent equipment per customer
    equipment_counts = df_quotes.groupby(
        ['numero_compte', 'regroup_famille_equipement_produit']
    ).size().reset_index(name='count')

    primary_equipment = equipment_counts.sort_values(
        ['numero_compte', 'count'], ascending=[True, False]
    ).drop_duplicates('numero_compte', keep='first')

    # Map complexity
    primary_equipment['equipment_complexity'] = primary_equipment[
        'regroup_famille_equipement_produit'
    ].map(equipment_complexity).fillna(2)

    # Brand loyalty
    brand_counts = df_quotes.groupby(['numero_compte', 'marque_produit']).size().reset_index(name='brand_count')
    total_counts = df_quotes.groupby('numero_compte').size().reset_index(name='total_quotes')

    brand_stats = pd.merge(brand_counts, total_counts, on='numero_compte')
    brand_stats['brand_proportion'] = brand_stats['brand_count'] / brand_stats['total_quotes']

    # Most frequent brand proportion = brand loyalty index
    max_brand = brand_stats.loc[brand_stats.groupby('numero_compte')['brand_proportion'].idxmax()]
    max_brand = max_brand.rename(columns={
        'marque_produit': 'primary_brand',
        'brand_proportion': 'brand_loyalty_index'
    })[['numero_compte', 'primary_brand', 'brand_loyalty_index']]

    # Merge results
    result = pd.merge(
        primary_equipment[['numero_compte', 'regroup_famille_equipement_produit', 'equipment_complexity']],
        max_brand,
        on='numero_compte',
        how='left'
    )

    return result


def create_process_commercial_features(df_quotes):
    """
    Create process adoption and commercial role features
    """
    # Process adoption
    process_stats = df_quotes.groupby('numero_compte').agg(
        total_quotes=('fg_nouveau_process_relance_devis', 'count'),
        new_process_count=('fg_nouveau_process_relance_devis', lambda x: (x == 1).sum()),
        old_process_count=('fg_nouveau_process_relance_devis', lambda x: (x == 0).sum())
    ).reset_index()

    process_stats['new_process_ratio'] = process_stats['new_process_count'] / process_stats['total_quotes']
    process_stats['process_consistency'] = (
            (process_stats['new_process_ratio'] == 1) |
            (process_stats['new_process_ratio'] == 0)
    ).astype(int)

    # Commercial role
    commercial_stats = df_quotes.groupby('numero_compte').agg(
        unique_commercials=('fonction_commercial', 'nunique'),
        is_senior_commercial=('fonction_commercial',
                              lambda x: x.str.contains('Responsable|Directeur|Manager|Chef|Senior',
                                                       case=False, na=False).any())
    ).reset_index()

    commercial_stats['is_senior_commercial'] = commercial_stats['is_senior_commercial'].astype(int)

    # Merge
    result = pd.merge(
        process_stats[['numero_compte', 'new_process_ratio', 'process_consistency']],
        commercial_stats[['numero_compte', 'unique_commercials', 'is_senior_commercial']],
        on='numero_compte',
        how='left'
    )

    return result


def create_agency_region_features(df_quotes):
    """
    Create agency and regional features
    """
    # Primary agency
    agency_counts = df_quotes.groupby(['numero_compte', 'nom_agence']).size().reset_index(name='count')
    primary_agency = agency_counts.sort_values(['numero_compte', 'count'], ascending=[True, False])
    primary_agency = primary_agency.drop_duplicates('numero_compte', keep='first')
    primary_agency = primary_agency.rename(columns={'nom_agence': 'primary_agency'})[
        ['numero_compte', 'primary_agency']
    ]

    # Primary region
    region_counts = df_quotes.groupby(['numero_compte', 'nom_region']).size().reset_index(name='count')
    primary_region = region_counts.sort_values(['numero_compte', 'count'], ascending=[True, False])
    primary_region = primary_region.drop_duplicates('numero_compte', keep='first')
    primary_region = primary_region.rename(columns={'nom_region': 'primary_region'})[
        ['numero_compte', 'primary_region']
    ]

    # Agency switching
    agency_switching = df_quotes.groupby('numero_compte').agg(
        agency_switches=('nom_agence', lambda x: x.nunique() - 1)
    ).reset_index()

    # Merge
    result = pd.merge(primary_agency, primary_region, on='numero_compte', how='left')
    result = pd.merge(result, agency_switching, on='numero_compte', how='left')

    return result


def create_price_features(df_quotes):
    """
    Create price-related features
    """
    # Ensure price columns are numeric
    price_cols = ['mt_ttc_apres_aide_devis', 'mt_ttc_avant_aide_devis', 'mt_apres_remise_ht_devis']

    for col in price_cols:
        if col in df_quotes.columns:
            df_quotes[col] = pd.to_numeric(df_quotes[col], errors='coerce')

    # Group by customer
    price_stats = df_quotes.groupby('numero_compte').agg(
        avg_price=('mt_ttc_apres_aide_devis', 'mean'),
        min_price=('mt_ttc_apres_aide_devis', 'min'),
        max_price=('mt_ttc_apres_aide_devis', 'max'),
        price_range=('mt_ttc_apres_aide_devis', lambda x: x.max() - x.min() if len(x) > 1 else 0),
        avg_discount=('mt_remise_exceptionnelle_ht', 'mean')
    ).reset_index()

    price_stats['price_volatility'] = price_stats['price_range'] / (price_stats['avg_price'] + 1)

    return price_stats


def create_decision_speed_features(df_quotes, cutoff_date=None, max_history_days=730):
    """
    ULTRA-FAST vectorized decision speed features (CUSTOMER-LEVEL)
    WITH LEAKAGE PROTECTION

    Parameters:
    -----------
    df_quotes : DataFrame
        Historical quotes data (read-only)
    cutoff_date : datetime or str
        Reference date for temporal cutoff (features use only data before this date)
    max_history_days : int
        Maximum lookback window in days (default: 2 years)

    Returns exactly one row per customer
    """
    print("=" * 80)
    print("CREATING DECISION SPEED FEATURES (CUSTOMER-LEVEL) - LEAKAGE SAFE")
    print("=" * 80)

    # Quick validation
    required_cols = ['numero_compte', 'dt_creation_devis']
    missing_cols = [col for col in required_cols if col not in df_quotes.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return pd.DataFrame(columns=['numero_compte'])

    # Make a working copy to avoid modifying original
    df = df_quotes.copy()

    # Apply temporal cutoff to prevent lookahead bias
    if cutoff_date:
        cutoff_date = pd.to_datetime(cutoff_date)
        df = df[pd.to_datetime(df['dt_creation_devis']) <= cutoff_date]
        print(f"📅 Applied temporal cutoff: {cutoff_date.date()}")

    # Apply maximum lookback window
    if cutoff_date and max_history_days:
        min_date = cutoff_date - pd.Timedelta(days=max_history_days)
        df = df[pd.to_datetime(df['dt_creation_devis']) >= min_date]
        print(f"📅 Maximum lookback: {max_history_days} days")

    if len(df) == 0:
        print("⚠️ No data after temporal filtering")
        return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing {len(df):,} quotes for {df['numero_compte'].nunique():,} customers")

    # 1. Sort once
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)
    df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'])

    # 2. SINGLE GROUPBY to get all sequences
    print("👥 Single groupby aggregation...")

    customer_groups = df.groupby('numero_compte')
    customer_ids = list(customer_groups.groups.keys())
    n_customers = len(customer_ids)

    print(f"  Processing {n_customers:,} customers")

    # 3. VECTORIZED FEATURE CALCULATION
    print("⚡ Vectorized feature calculation...")

    # Initialize arrays
    is_quick_decider = np.full(n_customers, 0.5, dtype=float)
    engagement_efficiency_score = np.full(n_customers, 0.5, dtype=float)
    decision_pattern_clarity = np.full(n_customers, 0.5, dtype=float)
    recent_engagement_score = np.full(n_customers, 0.5, dtype=float)  # New: recency weighting
    total_quotes_array = np.zeros(n_customers, dtype=int)
    avg_days_between_array = np.zeros(n_customers, dtype=float)

    # Process each customer
    for i, (customer_id, group) in enumerate(customer_groups):
        quotes_count = len(group)
        total_quotes_array[i] = quotes_count

        if quotes_count == 0:
            continue

        # Extract dates
        dates = group['dt_creation_devis'].sort_values()

        if quotes_count == 1:
            # Single quote customer
            is_quick_decider[i] = 0.7
            avg_days_between_array[i] = 0
            decision_pattern_clarity[i] = 0.8
            recent_engagement_score[i] = 1.0  # Single quote = recent by definition

        else:
            # Multi-quote customer
            # Calculate days between quotes (historical only)
            days_between = (dates.iloc[1:].values - dates.iloc[:-1].values) / np.timedelta64(1, 'D')
            avg_days_between = days_between.mean() if len(days_between) > 0 else 0
            avg_days_between_array[i] = avg_days_between

            # Calculate recency-weighted engagement
            if cutoff_date:
                days_since_last = (cutoff_date - dates.iloc[-1]).days
                recent_engagement_score[i] = np.exp(-days_since_last / 30)  # Decay over 30 days

            # Quick decider logic - based on historical pattern
            if avg_days_between < 7:
                is_quick_decider[i] = 0.8
            elif avg_days_between < 30:
                is_quick_decider[i] = 0.6
            else:
                is_quick_decider[i] = 0.3

            # Engagement efficiency - weighted by recency
            engagement_duration = (dates.iloc[-1] - dates.iloc[0]).days + 1
            if engagement_duration > 0:
                engagement_density = quotes_count / engagement_duration
                engagement_efficiency_score[i] = min(0.3 + engagement_density * 0.4, 0.9)

            # Decision pattern clarity
            if quotes_count >= 3:
                decision_pattern_clarity[i] = 0.8
            else:
                decision_pattern_clarity[i] = 0.4

    print("✅ Vectorized calculations complete")

    # 4. CREATE FINAL DATAFRAME
    print("📝 Creating final DataFrame...")

    result = pd.DataFrame({
        'numero_compte': customer_ids,
        'is_quick_decider': is_quick_decider,
        'engagement_efficiency_score': engagement_efficiency_score,
        'decision_pattern_clarity': decision_pattern_clarity,
        'recent_engagement_score': recent_engagement_score,
        '_total_quotes': total_quotes_array,
        '_avg_days_between': avg_days_between_array
    })

    print(f"\n✅ Created 4 decision speed features for {len(result):,} customers")

    return result


def create_interaction_features(df_quotes, cutoff_date=None, max_history_days=730):
    """
    ULTRA-FAST vectorized interaction features (CUSTOMER-LEVEL)
    WITH LEAKAGE PROTECTION

    Parameters:
    -----------
    df_quotes : DataFrame
        Historical quotes data (read-only)
    cutoff_date : datetime or str
        Reference date for temporal cutoff
    max_history_days : int
        Maximum lookback window in days
    """
    print("=" * 80)
    print("CREATING INTERACTION FEATURES (CUSTOMER-LEVEL) - LEAKAGE SAFE")
    print("=" * 80)

    # Quick validation
    required_cols = ['numero_compte', 'dt_creation_devis', 'marque_produit',
                     'regroup_famille_equipement_produit']
    missing_cols = [col for col in required_cols if col not in df_quotes.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return pd.DataFrame(columns=['numero_compte'])

    # Make a working copy and apply temporal constraints
    df = df_quotes.copy()

    # Apply temporal cutoff
    if cutoff_date:
        cutoff_date = pd.to_datetime(cutoff_date)
        df = df[pd.to_datetime(df['dt_creation_devis']) <= cutoff_date]
        print(f"📅 Applied temporal cutoff: {cutoff_date.date()}")

    # Apply maximum lookback window
    if cutoff_date and max_history_days:
        min_date = cutoff_date - pd.Timedelta(days=max_history_days)
        df = df[pd.to_datetime(df['dt_creation_devis']) >= min_date]
        print(f"📅 Maximum lookback: {max_history_days} days")

    if len(df) == 0:
        print("⚠️ No data after temporal filtering")
        return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing {len(df):,} quotes for {df['numero_compte'].nunique():,} customers")

    # 1. Sort once
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).reset_index(drop=True)
    df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'])

    # 2. Static mappings
    premium_brands = {'MITSUBISHI ELECTRIC', 'VIESSMANN', 'BOSCH', 'DE DIETRICH', 'BUDERUS'}
    equipment_complexity = {
        'STOVE': 1, 'Poêle': 1,
        'BOILER_GAS': 2, 'Chaudière': 2, 'BOILER_OIL': 2,
        'AIR_CONDITIONER': 2, 'Climatisation': 2,
        'HEAT_PUMP': 3, 'Pompe à chaleur': 3,
        'CONDENSING_BOILER': 3,
        'GEOTHERMAL': 4, 'GEOTHERMIE': 4,
        'SOLAR_THERMAL': 4, 'SOLAIRE_THERMIQUE': 4
    }

    # 3. SINGLE GROUPBY
    print("👥 Single groupby aggregation...")

    customer_groups = df.groupby('numero_compte')
    customer_ids = list(customer_groups.groups.keys())
    n_customers = len(customer_ids)

    print(f"  Processing {n_customers:,} customers")

    # 4. VECTORIZED CALCULATION
    print("⚡ Vectorized feature calculation...")

    # Initialize arrays
    quick_high_engagement = np.zeros(n_customers, dtype=int)
    single_quote_brand_loyal = np.zeros(n_customers, dtype=int)
    balanced_solution_seeker = np.zeros(n_customers, dtype=int)
    price_sensitivity_score = np.zeros(n_customers, dtype=float)  # New feature
    brand_preference_score = np.zeros(n_customers, dtype=float)  # New feature

    # Internal arrays
    total_quotes_array = np.zeros(n_customers, dtype=int)
    avg_days_between_array = np.zeros(n_customers, dtype=float)
    brand_loyalty_array = np.zeros(n_customers, dtype=float)
    solution_complexity_array = np.zeros(n_customers, dtype=float)

    # Process each customer
    for i, (customer_id, group) in enumerate(customer_groups):
        quotes_count = len(group)
        total_quotes_array[i] = quotes_count

        if quotes_count == 0:
            continue

        # Extract sequences
        dates = group['dt_creation_devis'].sort_values()
        brands = group['marque_produit'].tolist()
        equipment = group['regroup_famille_equipement_produit'].tolist()

        # Calculate basic metrics
        if quotes_count > 1:
            days_between = (dates.iloc[1:].values - dates.iloc[:-1].values) / np.timedelta64(1, 'D')
            avg_days_between_array[i] = days_between.mean() if len(days_between) > 0 else 0

        # Brand analysis
        brand_counts = {}
        premium_count = 0
        for brand in brands:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            if brand in premium_brands:
                premium_count += 1

        # Brand loyalty
        if brand_counts:
            max_count = max(brand_counts.values())
            brand_loyalty_array[i] = max_count / quotes_count

        # Brand preference score
        if quotes_count > 0:
            brand_preference_score[i] = premium_count / quotes_count

        # Equipment complexity
        complexity_scores = []
        for eq in equipment:
            score = equipment_complexity.get(eq, 2.0)
            complexity_scores.append(score)

        if complexity_scores:
            solution_complexity_array[i] = np.mean(complexity_scores)

        # Price sensitivity (if data available)
        if 'mt_ttc_apres_aide_devis' in group.columns and quotes_count > 0:
            prices = pd.to_numeric(group['mt_ttc_apres_aide_devis'], errors='coerce')
            valid_prices = prices.dropna()
            if len(valid_prices) > 0:
                price_std = valid_prices.std()
                price_mean = valid_prices.mean()
                if price_mean > 0:
                    price_sensitivity_score[i] = price_std / price_mean

        # Calculate interaction features
        if quotes_count > 1:
            # Quick + high engagement (based on historical pattern only)
            quick_high_engagement[i] = int(
                (avg_days_between_array[i] < 14) and
                (quotes_count >= 3)  # Need enough history to be meaningful
            )

        # Single quote + brand loyalty
        if quotes_count == 1:
            top_brand = max(brand_counts, key=brand_counts.get) if brand_counts else None
            is_premium = top_brand in premium_brands if top_brand else False
            single_quote_brand_loyal[i] = int(
                (brand_loyalty_array[i] > 0.8) and is_premium
            )

        # Balanced solution seeker
        if quotes_count >= 2:  # Need at least 2 quotes for this pattern
            balanced_solution_seeker[i] = int(
                (2.0 <= solution_complexity_array[i] <= 3.5) and
                (brand_loyalty_array[i] > 0.6)
            )

    print("✅ Vectorized calculations complete")

    # 5. CREATE FINAL DATAFRAME
    print("📝 Creating final DataFrame...")

    result = pd.DataFrame({
        'numero_compte': customer_ids,
        'quick_high_engagement': quick_high_engagement,
        'single_quote_brand_loyal': single_quote_brand_loyal,
        'balanced_solution_seeker': balanced_solution_seeker,
        'brand_preference_score': brand_preference_score,
        'price_sensitivity_score': np.clip(price_sensitivity_score, 0, 1)
    })

    print(f"\n✅ Created 5 interaction features for {len(result):,} customers")

    return result


def create_purchase_velocity_features(df_quotes, prediction_date=None, mode='training'):
    """
    FULLY VECTORIZED version with complete leakage protection
    """
    print("=" * 80)
    print(f"CREATING VECTORIZED PURCHASE VELOCITY FEATURES - {mode.upper()} MODE")
    print("=" * 80)

    # 1. DATA PREP WITH LEAKAGE PROTECTION
    df = df_quotes.copy()

    if 'dt_creation_devis' not in df.columns:
        print("⚠️ Missing date column")
        return pd.DataFrame(columns=['numero_compte'])

    # Convert dates
    df['quote_date'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

    # Apply strict temporal filtering
    if prediction_date:
        prediction_date = pd.to_datetime(prediction_date)

        # Determine cutoff based on mode
        if mode == 'training':
            # For training: include up to prediction date
            mask = df['quote_date'] <= prediction_date
        else:  # inference
            # For inference: strict, exclude prediction date
            mask = df['quote_date'] < prediction_date

        df = df[mask].copy()

        # Apply lookback window
        lookback_days = 1095 if mode == 'training' else 730
        min_date = prediction_date - pd.Timedelta(days=lookback_days)
        df = df[df['quote_date'] >= min_date]

        print(f"📅 {mode.title()} mode: {lookback_days // 365}-year lookback")

    if len(df) == 0:
        return pd.DataFrame(columns=['numero_compte'])

    print(f"📊 Processing {len(df):,} quotes for {df['numero_compte'].nunique():,} customers")

    # 2. FULLY VECTORIZED CALCULATIONS
    # Sort for sequence operations
    df = df.sort_values(['numero_compte', 'quote_date'])

    # Calculate sequence numbers within customers
    df['quote_seq'] = df.groupby('numero_compte').cumcount()

    # Calculate days between quotes using vectorized shift
    df['prev_quote_date'] = df.groupby('numero_compte')['quote_date'].shift(1)
    df['days_since_prev'] = (df['quote_date'] - df['prev_quote_date']).dt.days

    # 3. VECTORIZED AGGREGATION
    # Customer-level statistics using agg()
    velocity_stats = df.groupby('numero_compte').agg(
        total_quotes=('quote_date', 'count'),
        first_quote_date=('quote_date', 'min'),
        last_quote_date=('quote_date', 'max'),
        avg_days_between=('days_since_prev', 'mean'),
        std_days_between=('days_since_prev', 'std'),
        min_days_between=('days_since_prev', 'min'),
        max_days_between=('days_since_prev', 'max')
    ).reset_index()

    # 4. VECTORIZED FEATURE ENGINEERING
    # Engagement duration
    velocity_stats['engagement_duration'] = (
                                                    velocity_stats['last_quote_date'] - velocity_stats[
                                                'first_quote_date']
                                            ).dt.days + 1

    # Engagement density
    velocity_stats['engagement_density'] = (
            velocity_stats['total_quotes'] / velocity_stats['engagement_duration']
    ).fillna(0)

    # Optimal timing score (vectorized)
    velocity_stats['optimal_timing_score'] = (
            1 - (velocity_stats['avg_days_between'] - 31.7).abs() / 31.7
    ).clip(0, 1)

    # Quote frequency categories (vectorized)
    conditions = [
        velocity_stats['avg_days_between'] < 7,
        velocity_stats['avg_days_between'] < 15,
        velocity_stats['avg_days_between'] < 25,
        velocity_stats['avg_days_between'] < 40,
        velocity_stats['avg_days_between'] >= 40
    ]
    choices = [0.8, 0.6, 0.4, 0.7, 0.3]
    velocity_stats['quote_frequency_score'] = np.select(conditions, choices, default=0.5)

    # Decision urgency (vectorized)
    urgency_conditions = [
        velocity_stats['avg_days_between'] < 3,
        velocity_stats['avg_days_between'] < 7,
        velocity_stats['avg_days_between'] < 14,
        velocity_stats['avg_days_between'] < 30,
        velocity_stats['avg_days_between'] >= 30
    ]
    urgency_choices = [0.9, 0.7, 0.5, 0.3, 0.1]
    velocity_stats['decision_urgency_score'] = np.select(urgency_conditions, urgency_choices, default=0.5)

    # Engagement consistency (vectorized)
    velocity_stats['engagement_consistency'] = (
            1 - (velocity_stats['std_days_between'] / (velocity_stats['avg_days_between'] + 1))
    ).clip(0, 1).fillna(1)  # NaN for single quotes = consistent

    # Tire-kicker indicator (vectorized)
    velocity_stats['tire_kicker_indicator'] = (
            ((velocity_stats['avg_days_between'] >= 15) &
             (velocity_stats['avg_days_between'] <= 25)).astype(float) * 0.4 +
            ((velocity_stats['std_days_between'] > velocity_stats['avg_days_between'] * 0.5) &
             (velocity_stats['total_quotes'] > 1)).astype(float) * 0.3 +
            (velocity_stats['total_quotes'] > 3).astype(float) * 0.3
    ).clip(0, 1)

    # Brief engagement risk (vectorized)
    velocity_stats['brief_engagement_risk'] = (
            (velocity_stats['avg_days_between'] < 7).astype(float) * 0.6 +
            (velocity_stats['total_quotes'] == 2).astype(float) * 0.4
    ).clip(0, 1)

    # Purchase velocity index (vectorized composite)
    velocity_stats['purchase_velocity_index'] = (
            velocity_stats['optimal_timing_score'] * 0.4 +
            velocity_stats['decision_urgency_score'] * 0.3 +
            velocity_stats['engagement_consistency'] * 0.3
    ).clip(0, 1)

    # Recency score if prediction_date available
    if prediction_date:
        velocity_stats['days_since_last_quote'] = (
                prediction_date - velocity_stats['last_quote_date']
        ).dt.days
        velocity_stats['recency_score'] = np.exp(-velocity_stats['days_since_last_quote'] / 30).clip(0, 1)

    # Select final columns
    result_cols = ['numero_compte', 'quote_frequency_score', 'decision_urgency_score',
                   'engagement_consistency', 'optimal_timing_score',
                   'tire_kicker_indicator', 'brief_engagement_risk',
                   'purchase_velocity_index']

    if 'recency_score' in velocity_stats.columns:
        result_cols.append('recency_score')

    print(f"✅ Created {len(result_cols) - 1} vectorized features for {len(velocity_stats):,} customers")

    return velocity_stats[result_cols]


# Also need to fix the vectorized version to match your column names:
def create_decision_consistency_features(df_quotes, prediction_date=None, mode='training'):
    """
    Create features related to decision consistency and focus.
    Fixed version with proper column creation.
    """
    print("=" * 80)
    print(f"CREATING DECISION CONSISTENCY FEATURES - {mode.upper()} MODE")
    print("=" * 80)

    # Data preparation
    df = df_quotes.copy()

    # Apply temporal filtering if date exists
    if prediction_date and 'dt_creation_devis' in df.columns:
        df['quote_date'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')
        prediction_date = pd.to_datetime(prediction_date)

        if mode == 'training':
            df = df[df['quote_date'] <= prediction_date]
        else:  # inference
            df = df[df['quote_date'] < prediction_date]

    if len(df) == 0:
        return pd.DataFrame(columns=['numero_compte'])

    # Equipment complexity mapping
    equipment_complexity = {
        'STOVE': 1, 'Poêle': 1,
        'BOILER_GAS': 2, 'Chaudière': 2, 'BOILER_OIL': 2,
        'AIR_CONDITIONER': 2, 'Climatisation': 2,
        'HEAT_PUMP': 3, 'Pompe à chaleur': 3,
        'CONDENSING_BOILER': 3,
        'GEOTHERMAL': 4, 'GEOTHERMIE': 4,
        'SOLAR_THERMAL': 4, 'SOLAIRE_THERMIQUE': 4
    }

    # Create equipment_complexity column BEFORE groupby
    if 'regroup_famille_equipement_produit' in df.columns:
        df['equipment_complexity'] = df['regroup_famille_equipement_produit'].map(
            lambda x: equipment_complexity.get(x, 2.0)
        )

    # Define a safe function for groupby
    def get_group_stats(group):
        stats = {
            'total_quotes': len(group),
        }

        # Price statistics
        if 'mt_ttc_apres_aide_devis' in group.columns:
            prices = pd.to_numeric(group['mt_ttc_apres_aide_devis'], errors='coerce')
            valid_prices = prices.dropna()
            if len(valid_prices) > 0:
                stats['avg_price'] = valid_prices.mean()
                stats['price_std'] = valid_prices.std()
                stats['price_min'] = valid_prices.min()
                stats['price_max'] = valid_prices.max()
            else:
                stats['avg_price'] = np.nan
                stats['price_std'] = np.nan
                stats['price_min'] = np.nan
                stats['price_max'] = np.nan

        # Equipment statistics
        if 'regroup_famille_equipement_produit' in group.columns:
            stats['unique_equipment'] = group['regroup_famille_equipement_produit'].nunique()

        # Equipment complexity statistics
        if 'equipment_complexity' in group.columns:
            stats['avg_complexity'] = group['equipment_complexity'].mean()
            if len(group) > 1:
                stats['complexity_std'] = group['equipment_complexity'].std()
            else:
                stats['complexity_std'] = 0

        # Brand statistics
        if 'marque_produit' in group.columns:
            brand_counts = group['marque_produit'].value_counts()
            if not brand_counts.empty:
                stats['top_brand_count'] = brand_counts.iloc[0]
                stats['unique_brands'] = len(brand_counts)
            else:
                stats['top_brand_count'] = 0
                stats['unique_brands'] = 0

        # Process statistics
        if 'fg_nouveau_process_relance_devis' in group.columns:
            stats['new_process_count'] = (group['fg_nouveau_process_relance_devis'] == 1).sum()

        return pd.Series(stats)

    # Apply groupby
    result = df.groupby('numero_compte').apply(get_group_stats).reset_index()

    # Fill missing columns with defaults
    required_columns = ['avg_price', 'price_std', 'price_min', 'price_max',
                        'unique_equipment', 'avg_complexity', 'complexity_std',
                        'top_brand_count', 'unique_brands', 'new_process_count']

    for col in required_columns:
        if col not in result.columns:
            if col in ['avg_price', 'price_std', 'price_min', 'price_max', 'avg_complexity', 'complexity_std']:
                result[col] = np.nan
            else:
                result[col] = 0

    # Calculate features
    # Price consistency
    result['price_consistency_score'] = (1 - (result['price_std'] / (result['avg_price'] + 1))).clip(0, 1).fillna(0.5)

    # Equipment focus
    result['equipment_focus_score'] = (1 - (result['unique_equipment'] / result['total_quotes'])).clip(0, 1).fillna(0.5)

    # Brand loyalty
    result['brand_loyalty_score'] = (result['top_brand_count'] / result['total_quotes']).clip(0, 1).fillna(0.5)

    # Solution stability
    result['solution_stability_score'] = (1 - (result['complexity_std'] / 2)).clip(0, 1).fillna(0.5)

    # Budget clarity
    result['price_range'] = result['price_max'] - result['price_min']
    result['budget_clarity_score'] = np.select(
        [
            result['price_range'] / (result['avg_price'] + 1) < 0.1,
            result['price_range'] / (result['avg_price'] + 1) < 0.3,
            result['price_range'] / (result['avg_price'] + 1) < 0.5,
            True
        ],
        [0.9, 0.7, 0.4, 0.2],
        default=0.5
    )

    # New process ratio
    result['new_process_ratio'] = (result['new_process_count'] / result['total_quotes']).fillna(0)

    # Process consistency
    result['process_consistency'] = ((result['new_process_ratio'] == 1) | (result['new_process_ratio'] == 0)).astype(
        float)

    # Needs refinement
    result['needs_refinement_indicator'] = (
            (result['price_consistency_score'] < 0.6).astype(float) * 0.3 +
            (result['unique_equipment'] >= 3).astype(float) * 0.2 +
            (result['complexity_std'].fillna(0) >= 1.0).astype(float) * 0.3 +
            (result['budget_clarity_score'] <= 0.2).astype(float) * 0.2
    ).clip(0, 1)

    # Decision maturity
    result['decision_maturity_index'] = (
            result['price_consistency_score'] * 0.25 +
            result['equipment_focus_score'] * 0.2 +
            result['brand_loyalty_score'] * 0.2 +
            result['solution_stability_score'] * 0.2 +
            result['budget_clarity_score'] * 0.15
    ).clip(0, 1)

    # Select final columns
    final_cols = [
        'numero_compte', 'price_consistency_score', 'equipment_focus_score',
        'brand_loyalty_score', 'solution_stability_score', 'budget_clarity_score',
        'process_consistency', 'needs_refinement_indicator', 'decision_maturity_index'
    ]

    print(f"✅ Created {len(final_cols) - 1} features for {len(result):,} customers")
    return result[final_cols]


def create_error_pattern_interactions(all_features):
    """
    Create interaction features that specifically target identified error patterns.
    Fixed version with proper all() usage.
    """
    print("  → Creating error pattern interactions...")

    interactions = pd.DataFrame({
        'numero_compte': all_features['numero_compte']
    })

    # ----------------------------------------------------
    # 1. FALSE POSITIVE TARGETING (Tire-kicker patterns)
    # ----------------------------------------------------

    # FIXED: Use set.issubset() directly or check each column
    if {'tire_kicker_indicator', 'engagement_density'}.issubset(set(all_features.columns)):
        # Pattern: High engagement density but tire-kicker indicator = wasted effort
        interactions['wasted_effort_risk'] = (
                all_features['tire_kicker_indicator'] *
                all_features['engagement_density']
        )
        print("    Created: wasted_effort_risk (targets False Positives)")

    # FIXED: Simplified check for two columns
    if ('brief_engagement_risk' in all_features.columns and
            'decision_urgency_score' in all_features.columns):
        # Pattern: Brief engagement + high urgency = quick converter potential
        interactions['quick_converter_potential'] = (
                all_features['brief_engagement_risk'] *
                all_features['decision_urgency_score']
        )
        print("    Created: quick_converter_potential (targets False Negatives)")

    # ----------------------------------------------------
    # 2. CONSISTENCY-VALUE INTERACTIONS
    # ----------------------------------------------------
    if ('price_consistency_score' in all_features.columns and
            'avg_price' in all_features.columns):
        # Pattern: Consistent prices + high value = serious high-value buyer
        avg_price_quantile = all_features['avg_price'].quantile(0.75)
        if avg_price_quantile > 0:
            interactions['serious_high_value_buyer'] = (
                    all_features['price_consistency_score'] *
                    (all_features['avg_price'] / avg_price_quantile)
            ).clip(0, 1)
            print("    Created: serious_high_value_buyer (targets True Positives)")

    if ('solution_stability_score' in all_features.columns and
            'equipment_complexity' in all_features.columns):
        # Pattern: Stable solution + high complexity = serious complex solution seeker
        interactions['complex_solution_seeker'] = (
                all_features['solution_stability_score'] *
                (all_features['equipment_complexity'] / 4)  # Normalize to 0-1
        ).clip(0, 1)
        print("    Created: complex_solution_seeker (targets True Positives)")

    # ----------------------------------------------------
    # 3. TIMING-CONSISTENCY INTERACTIONS
    # ----------------------------------------------------
    if ('optimal_timing_score' in all_features.columns and
            'engagement_consistency' in all_features.columns):
        # Pattern: Optimal timing + consistent engagement = ideal buyer pattern
        interactions['ideal_buyer_pattern'] = (
                all_features['optimal_timing_score'] *
                all_features['engagement_consistency']
        )
        print("    Created: ideal_buyer_pattern (enhances True Positives)")

    if ('quote_frequency_score' in all_features.columns and
            'brand_loyalty_score' in all_features.columns):
        # Pattern: Right frequency + brand loyalty = loyal repeat buyer potential
        interactions['loyal_repeat_potential'] = (
                all_features['quote_frequency_score'] *
                all_features['brand_loyalty_score']
        )
        print("    Created: loyal_repeat_potential (enhances True Positives)")

    # ----------------------------------------------------
    # 4. BUSINESS LOGIC INTERACTIONS
    # ----------------------------------------------------
    if ('needs_refinement_indicator' in all_features.columns and
            'total_quotes' in all_features.columns):
        # Pattern: Needs refinement + many quotes = indecisive shopper
        max_total_quotes = all_features['total_quotes'].max()
        if max_total_quotes > 0:
            interactions['indecisive_shopper'] = (
                    all_features['needs_refinement_indicator'] *
                    np.log1p(all_features['total_quotes']) / np.log1p(max_total_quotes)
            ).clip(0, 1)
            print("    Created: indecisive_shopper (targets False Positives)")

    if ('budget_clarity_score' in all_features.columns and
            'price_volatility' in all_features.columns):
        # Pattern: Clear budget + low price volatility = ready to purchase
        interactions['purchase_ready'] = (
                all_features['budget_clarity_score'] *
                (1 - all_features['price_volatility'].clip(0, 1))
        )
        print("    Created: purchase_ready (targets True Positives)")

    print(f"    Total: Created {len(interactions.columns)} error pattern interactions")
    return interactions


def create_business_ready_scores(all_features):
    """
    Create business-ready composite scores for sales team action
    Fixed version with proper column checks.
    """
    print("  → Creating business-ready scores...")

    scores = pd.DataFrame({
        'numero_compte': all_features['numero_compte']
    })

    # ----------------------------------------------------
    # 1. SALES PRIORITY SCORE (Who to contact first)
    # ----------------------------------------------------
    priority_components = []
    priority_weights = []

    # High priority: Brief engagement risk
    if 'brief_engagement_risk' in all_features.columns:
        priority_components.append(all_features['brief_engagement_risk'])
        priority_weights.append(0.30)
        print("    Priority component: brief_engagement_risk (weight: 0.30)")

    # High priority: Not a tire-kicker
    if 'tire_kicker_indicator' in all_features.columns:
        priority_components.append(1 - all_features['tire_kicker_indicator'])
        priority_weights.append(0.25)
        print("    Priority component: inverse_tire_kicker (weight: 0.25)")

    # Medium priority: Decision urgency
    if 'decision_urgency_score' in all_features.columns:
        priority_components.append(all_features['decision_urgency_score'])
        priority_weights.append(0.20)
        print("    Priority component: decision_urgency_score (weight: 0.20)")

    # Medium priority: Budget clarity
    if 'budget_clarity_score' in all_features.columns:
        priority_components.append(all_features['budget_clarity_score'])
        priority_weights.append(0.15)
        print("    Priority component: budget_clarity_score (weight: 0.15)")

    # Low priority: Recent engagement
    if 'recent_engagement_score' in all_features.columns:
        priority_components.append(all_features['recent_engagement_score'])
        priority_weights.append(0.10)
        print("    Priority component: recent_engagement_score (weight: 0.10)")

    if priority_components:
        # Combine all components with weights
        weighted_sum = sum(comp * weight for comp, weight in zip(priority_components, priority_weights))
        total_weight = sum(priority_weights)
        scores['sales_priority_score'] = (weighted_sum / total_weight).clip(0, 1)
        print("    Created: sales_priority_score (for sales team action)")

    # ----------------------------------------------------
    # 2. CONVERSION CONFIDENCE SCORE
    # ----------------------------------------------------
    confidence_components = []
    confidence_weights = []

    if 'solution_stability_score' in all_features.columns:
        confidence_components.append(all_features['solution_stability_score'])
        confidence_weights.append(0.25)
        print("    Confidence component: solution_stability_score (weight: 0.25)")

    if 'brand_loyalty_score' in all_features.columns:
        confidence_components.append(all_features['brand_loyalty_score'])
        confidence_weights.append(0.20)
        print("    Confidence component: brand_loyalty_score (weight: 0.20)")

    if 'engagement_consistency' in all_features.columns:
        confidence_components.append(all_features['engagement_consistency'])
        confidence_weights.append(0.20)
        print("    Confidence component: engagement_consistency (weight: 0.20)")

    if 'price_consistency_score' in all_features.columns:
        confidence_components.append(all_features['price_consistency_score'])
        confidence_weights.append(0.35)
        print("    Confidence component: price_consistency_score (weight: 0.35)")

    if confidence_components:
        weighted_sum = sum(comp * weight for comp, weight in zip(confidence_components, confidence_weights))
        total_weight = sum(confidence_weights)
        scores['conversion_confidence_score'] = (weighted_sum / total_weight).clip(0, 1)
        print("    Created: conversion_confidence_score (model confidence level)")

    # ----------------------------------------------------
    # 3. BUYER READINESS SCORE
    # ----------------------------------------------------
    readiness_components = []
    readiness_weights = []

    if 'optimal_timing_score' in all_features.columns:
        readiness_components.append(all_features['optimal_timing_score'])
        readiness_weights.append(0.25)
        print("    Readiness component: optimal_timing_score (weight: 0.25)")

    if 'decision_maturity_index' in all_features.columns:
        readiness_components.append(all_features['decision_maturity_index'])
        readiness_weights.append(0.20)
        print("    Readiness component: decision_maturity_index (weight: 0.20)")

    if 'purchase_velocity_index' in all_features.columns:
        readiness_components.append(all_features['purchase_velocity_index'])
        readiness_weights.append(0.20)
        print("    Readiness component: purchase_velocity_index (weight: 0.20)")

    if 'equipment_focus_score' in all_features.columns:
        readiness_components.append(all_features['equipment_focus_score'])
        readiness_weights.append(0.15)
        print("    Readiness component: equipment_focus_score (weight: 0.15)")

    if 'quote_frequency_score' in all_features.columns:
        readiness_components.append(all_features['quote_frequency_score'])
        readiness_weights.append(0.10)
        print("    Readiness component: quote_frequency_score (weight: 0.10)")

    if 'needs_refinement_indicator' in all_features.columns:
        readiness_components.append(1 - all_features['needs_refinement_indicator'])
        readiness_weights.append(0.10)
        print("    Readiness component: inverse_needs_refinement (weight: 0.10)")

    if readiness_components:
        weighted_sum = sum(comp * weight for comp, weight in zip(readiness_components, readiness_weights))
        total_weight = sum(readiness_weights)
        scores['buyer_readiness_score'] = (weighted_sum / total_weight).clip(0, 1)
        print("    Created: buyer_readiness_score (overall purchase readiness)")

    print(f"    Total: Created {len(scores.columns)} business-ready scores")
    return scores


def enhance_existing_features_with_dominant_signals(existing_features, mode='standard'):
    """
    Add dominant behavioral signals to existing customer-level features
    mode: 'standard' or 'ultra' for different feature sets
    """
    enhanced = existing_features.copy()
    new_features_added = 0

    # ============================================
    # STANDARD DOMINANT FEATURES (always add these)
    # ============================================

    # 1. Quote efficiency (combines quote count with decision quality)
    if ('total_quotes' in enhanced.columns and
            'decision_maturity_index' in enhanced.columns and
            'price_consistency_score' in enhanced.columns):
        enhanced['quote_efficiency_dominant'] = (
                np.log1p(enhanced['total_quotes']) *
                (enhanced['decision_maturity_index'] * 0.6 +
                 enhanced['price_consistency_score'] * 0.4)
        )
        new_features_added += 1

    # 2. Engagement pattern anomaly (direct FP/TP encoding)
    if 'avg_days_between' in enhanced.columns:
        enhanced['engagement_pattern_anomaly'] = np.select(
            [
                (enhanced['avg_days_between'] >= 17) & (enhanced['avg_days_between'] <= 27),
                (enhanced['avg_days_between'] >= 27) & (enhanced['avg_days_between'] <= 37),
                True
            ],
            [0.8, 0.2, 0.5],
            default=0.5
        )
        new_features_added += 1

    # 3. Is tire-kicker? (binary, impossible to ignore)
    if ('tire_kicker_indicator' in enhanced.columns and
            'engagement_density' in enhanced.columns):
        enhanced['is_tire_kicker_binary'] = (
                (enhanced['tire_kicker_indicator'] > 0.7) |
                (enhanced['engagement_density'] > 1.0)
        ).astype(float)
        new_features_added += 1

    # 4. Days from optimal (normalized)
    if 'avg_days_between' in enhanced.columns:
        enhanced['days_from_optimal_normalized'] = enhanced['avg_days_between'] / 31.6
        new_features_added += 1

    # ============================================
    # ULTRA-DOMINANT FEATURES (optional, more aggressive)
    # ============================================

    if mode == 'ultra':
        print("  → Adding ULTRA-dominant features...")

        # 5. ULTRA: Exact tire-kicker match (from your error analysis: 21.122)
        if 'avg_days_between' in enhanced.columns:
            enhanced['exact_tire_kicker_match'] = (
                # High score if EXACTLY in FP range (21.122 ± 2)
                    ((enhanced['avg_days_between'] >= 19) &
                     (enhanced['avg_days_between'] <= 23)).astype(float) * 0.9 +
                    # Medium score if close to FP range
                    ((enhanced['avg_days_between'] >= 16) &
                     (enhanced['avg_days_between'] <= 26)).astype(float) * 0.6
            ).clip(0, 1)
            new_features_added += 1

        # 6. ULTRA: Exact serious buyer match (from your analysis: 31.455)
        if 'avg_days_between' in enhanced.columns:
            enhanced['exact_serious_buyer_match'] = (
                # High score if EXACTLY in TP range (31.455 ± 3)
                    ((enhanced['avg_days_between'] >= 28) &
                     (enhanced['avg_days_between'] <= 35)).astype(float) * 0.9 +
                    # Medium score if close to TP range
                    ((enhanced['avg_days_between'] >= 26) &
                     (enhanced['avg_days_between'] <= 37)).astype(float) * 0.6
            ).clip(0, 1)
            new_features_added += 1

        # 7. ULTRA: Engagement efficiency ratio (mathematically optimal)
        if all({'avg_days_between', 'engagement_density'}.issubset(enhanced.columns)):
            # FPs: 21.122 days, 0.893 density
            # TPs: 31.455 days, 0.811 density
            enhanced['engagement_efficiency_ratio'] = (
                    (enhanced['avg_days_between'] / 31.455) *  # Days from optimal (TP)
                    (0.811 / (enhanced['engagement_density'] + 0.001))  # Inverse density penalty
            )
            # Normalize to 0-1
            q95 = enhanced['engagement_efficiency_ratio'].quantile(0.95)
            if q95 > 0:
                enhanced['engagement_efficiency_ratio'] = (
                        enhanced['engagement_efficiency_ratio'] / q95
                ).clip(0, 1)
            new_features_added += 1

        # 8. ULTRA: Binary classifier in feature form
        if all({'avg_days_between', 'engagement_density', 'price_consistency_score'}.issubset(enhanced.columns)):
            enhanced['binary_tire_kicker_classifier'] = (
                # If all 3 conditions point to tire-kicker
                    ((enhanced['avg_days_between'] < 25).astype(float) * 0.4) +
                    ((enhanced['engagement_density'] > 0.85).astype(float) * 0.3) +
                    ((enhanced['price_consistency_score'] < 0.7).astype(float) * 0.3)
            ).clip(0, 1)
            new_features_added += 1

    # ============================================
    # FINAL: Add interaction between standard and ultra features
    # ============================================

    if mode == 'ultra' and 'quote_efficiency_dominant' in enhanced.columns:
        if 'exact_serious_buyer_match' in enhanced.columns:
            enhanced['super_serious_buyer'] = (
                    enhanced['quote_efficiency_dominant'] *
                    enhanced['exact_serious_buyer_match']
            )
            new_features_added += 1

        if 'exact_tire_kicker_match' in enhanced.columns:
            enhanced['super_tire_kicker'] = (
                    enhanced['quote_efficiency_dominant'] *
                    (1 - enhanced['exact_tire_kicker_match'])  # Inverse: high = NOT tire-kicker
            )
            new_features_added += 1

    print(f"  → Added {new_features_added} dominant features ({mode} mode)")

    # Show which features were added
    if new_features_added > 0:
        new_cols = [col for col in enhanced.columns if col not in existing_features.columns]
        print(f"    New: {', '.join(new_cols[:8])}" +
              ("..." if len(new_cols) > 8 else ""))

    return enhanced


def create_correction_features(df_quotes, prediction_date=None, mode='training'):
    """
    Create all features from raw quote data with error-correction enhancements
    CREATE ALL FEATURES FIRST, THEN MERGE ONCE
    """
    print("Creating features from raw data with error correction...")

    # 1. CREATE ALL FEATURE SETS FIRST
    print("  → Creating ALL feature sets...")

    # Create each feature set independently
    engagement = create_engagement_features(df_quotes)
    equipment = create_equipment_brand_features(df_quotes)
    process = create_process_commercial_features(df_quotes)
    agency = create_agency_region_features(df_quotes)
    price = create_price_features(df_quotes)
    decision_speed = create_decision_speed_features(df_quotes, prediction_date)
    interaction = create_interaction_features(df_quotes, prediction_date)
    velocity = create_purchase_velocity_features(df_quotes, prediction_date, mode)
    consistency = create_decision_consistency_features(df_quotes, prediction_date, mode)

    # 2. CREATE INTERACTIONS AND SCORES FROM RELEVANT FEATURES
    print("  → Creating interactions and scores...")

    # Merge only what's needed for interactions
    temp_for_interactions = pd.merge(engagement, velocity, on='numero_compte', how='left')
    temp_for_interactions = pd.merge(temp_for_interactions, consistency, on='numero_compte', how='left')
    temp_for_interactions = pd.merge(temp_for_interactions, price, on='numero_compte', how='left')

    # Now create the interactions and scores
    error_interactions = create_error_pattern_interactions(temp_for_interactions)
    business_scores = create_business_ready_scores(temp_for_interactions)

    # 3. SINGLE MERGE OF EVERYTHING
    print("  → SINGLE MERGE of all features...")

    # List of ALL feature DataFrames to merge
    all_feature_sets = [
        engagement,
        equipment,
        process,
        agency,
        price,
        decision_speed,
        interaction,
        velocity,
        consistency,
        error_interactions,
        business_scores
    ]

    # Start with the first feature set
    all_features = engagement

    # SINGLE LOOP - SINGLE MERGE OPERATION
    for feature_set in all_feature_sets[1:]:  # Skip first since it's our starting point
        if not feature_set.empty:
            all_features = pd.merge(all_features, feature_set, on='numero_compte', how='left')

    # 4. Fill missing values
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns
    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)

    # 5. Add target if available
    if 'fg_devis_accepte' in df_quotes.columns:
        print("  → Adding conversion target...")
        conversion_status = df_quotes.groupby('numero_compte')['fg_devis_accepte'].max().reset_index()
        conversion_status = conversion_status.rename(columns={'fg_devis_accepte': 'converted'})
        all_features = pd.merge(all_features, conversion_status, on='numero_compte', how='left')

    print(f"✅ Created {len(all_features.columns) - 1} features for {len(all_features)} customers")

    final_features = enhance_existing_features_with_dominant_signals(all_features)

    return final_features