import pandas as pd
import numpy as np


def create_process_features(
        df_quotes: pd.DataFrame,
        cutoff_date: str = None,
        customer_col: str = "numero_compte",
        process_col: str = "fg_nouveau_process_relance_devis",
        target_col: str = "fg_devis_accepte"
) -> pd.DataFrame:
    """
    CUSTOMER-LEVEL process features with leakage protection
    Returns exactly one row per customer

    Parameters:
    -----------
    df_quotes : DataFrame
        Historical quotes data
    cutoff_date : str or datetime
        Reference date for temporal filtering (prevents lookahead bias)
    customer_col : str
        Customer identifier column
    process_col : str
        New process flag column
    target_col : str
        Target column (if available for debugging)

    Returns:
    --------
    DataFrame with one row per customer and process-related features
    """
    print("=" * 80)
    print("CREATING CUSTOMER-LEVEL PROCESS FEATURES - LEAKAGE SAFE")
    print("=" * 80)

    # Validate required columns
    required_cols = [customer_col, process_col]
    missing_cols = [col for col in required_cols if col not in df_quotes.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return pd.DataFrame(columns=[customer_col])

    # Make a working copy
    df = df_quotes.copy()

    # Apply temporal cutoff if provided
    if cutoff_date and 'dt_creation_devis' in df.columns:
        cutoff_date = pd.to_datetime(cutoff_date)
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')
        df = df[df['dt_creation_devis'] <= cutoff_date]
        print(f"📅 Applied temporal cutoff: {cutoff_date.date()}")
        print(f"📊 Quotes after filtering: {len(df):,}")

    if len(df) == 0:
        print("⚠️ No data after temporal filtering")
        return pd.DataFrame(columns=[customer_col])

    print(f"Processing {len(df):,} quotes for {df[customer_col].nunique():,} customers")

    # Sort by customer and date if available
    sort_keys = [customer_col]
    if 'dt_creation_devis' in df.columns:
        sort_keys.append('dt_creation_devis')
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # --------------------------------------------------------------------
    # 1. GROUP BY CUSTOMER - SINGLE PASS FOR ALL FEATURES
    # --------------------------------------------------------------------
    print("👥 Grouping by customer...")

    customer_groups = df.groupby(customer_col)
    customer_ids = list(customer_groups.groups.keys())
    n_customers = len(customer_ids)

    print(f"  Processing {n_customers:,} customers")

    # --------------------------------------------------------------------
    # 2. INITIALIZE FEATURE ARRAYS
    # --------------------------------------------------------------------
    print("⚡ Calculating customer-level features...")

    # Basic metrics
    total_quotes = np.zeros(n_customers, dtype=int)
    process_adoption_rate = np.zeros(n_customers, dtype=float)
    process_consistency = np.zeros(n_customers, dtype=float)
    recent_process_adoption = np.zeros(n_customers, dtype=float)
    process_volatility = np.zeros(n_customers, dtype=float)

    # Binary flags
    has_ever_used_process = np.zeros(n_customers, dtype=int)
    process_preference = np.zeros(n_customers, dtype=int)  # 0=never, 1=sometimes, 2=always
    current_process_usage = np.zeros(n_customers, dtype=int)

    # Temporal metrics (if dates available)
    if 'dt_creation_devis' in df.columns:
        days_since_first_quote = np.zeros(n_customers, dtype=float)
        days_since_last_quote = np.zeros(n_customers, dtype=float)
        process_adoption_trend = np.zeros(n_customers, dtype=float)

    # --------------------------------------------------------------------
    # 3. PROCESS EACH CUSTOMER
    # --------------------------------------------------------------------
    for i, (customer_id, group) in enumerate(customer_groups):
        quotes_count = len(group)
        total_quotes[i] = quotes_count

        if quotes_count == 0:
            continue

        # Extract process flags
        process_flags = group[process_col]

        # Handle missing values
        valid_process_flags = process_flags.dropna()
        if len(valid_process_flags) == 0:
            process_adoption_rate[i] = 0
            has_ever_used_process[i] = 0
            process_consistency[i] = 1  # Consistent in not using
            continue

        # Convert to binary (1 = new process, 0 = old process)
        # Assuming process_col = 1 means new process, 0 means old
        is_new_process = (valid_process_flags == 1).astype(int)

        # Calculate adoption rate
        adoption_rate = is_new_process.mean()
        process_adoption_rate[i] = adoption_rate

        # Has ever used new process
        has_ever_used_process[i] = 1 if is_new_process.sum() > 0 else 0

        # Process consistency (how consistent is the process choice)
        if len(is_new_process) >= 2:
            changes = np.abs(np.diff(is_new_process)).sum()
            process_consistency[i] = 1 - (changes / (len(is_new_process) - 1))
        else:
            process_consistency[i] = 1  # Single quote = consistent

        # Process volatility (standard deviation)
        if len(is_new_process) >= 2:
            process_volatility[i] = is_new_process.std()
        else:
            process_volatility[i] = 0

        # Process preference categories
        if adoption_rate == 0:
            process_preference[i] = 0  # Never uses new process
        elif adoption_rate == 1:
            process_preference[i] = 2  # Always uses new process
        else:
            process_preference[i] = 1  # Sometimes uses new process

        # Current process usage (latest quote)
        current_process_usage[i] = is_new_process.iloc[-1] if len(is_new_process) > 0 else 0

        # Recent process adoption (last 3 quotes if available)
        if len(is_new_process) >= 3:
            recent_adoption = is_new_process.iloc[-3:].mean()
            recent_process_adoption[i] = recent_adoption
        elif len(is_new_process) > 0:
            recent_process_adoption[i] = adoption_rate

        # Temporal features if dates available
        if 'dt_creation_devis' in df.columns:
            dates = pd.to_datetime(group['dt_creation_devis']).sort_values()

            if cutoff_date:
                # Days since first and last quote
                days_since_first_quote[i] = (cutoff_date - dates.iloc[0]).days
                days_since_last_quote[i] = (cutoff_date - dates.iloc[-1]).days

            # Process adoption trend (if enough quotes)
            if len(is_new_process) >= 3:
                # Split into early vs late adoption
                split_point = len(is_new_process) // 2
                early_adoption = is_new_process.iloc[:split_point].mean()
                late_adoption = is_new_process.iloc[split_point:].mean()
                process_adoption_trend[i] = late_adoption - early_adoption

    # --------------------------------------------------------------------
    # 4. CREATE FINAL DATAFRAME
    # --------------------------------------------------------------------
    print("📝 Creating final DataFrame...")

    # Base features for all customers
    feature_dict = {
        customer_col: customer_ids,
        'total_quotes': total_quotes,
        'process_adoption_rate': process_adoption_rate,
        'has_ever_used_process': has_ever_used_process,
        'process_consistency': process_consistency,
        'process_volatility': np.clip(process_volatility, 0, 1),
        'process_preference': process_preference,
        'current_process_usage': current_process_usage,
        'recent_process_adoption': np.clip(recent_process_adoption, 0, 1)
    }

    # Add temporal features if dates available
    if 'dt_creation_devis' in df.columns and cutoff_date:
        feature_dict.update({
            'days_since_first_quote': days_since_first_quote,
            'days_since_last_quote': days_since_last_quote,
            'process_adoption_trend': np.clip(process_adoption_trend, -1, 1)
        })

    result = pd.DataFrame(feature_dict)

    # --------------------------------------------------------------------
    # 5. CREATE DERIVED FEATURES
    # --------------------------------------------------------------------

    # Process engagement intensity (normalized by time if dates available)
    if 'days_since_first_quote' in result.columns:
        # Avoid division by zero
        mask = result['days_since_first_quote'] > 0
        result.loc[mask, 'process_engagement_intensity'] = (
                result.loc[mask, 'total_quotes'] / result.loc[mask, 'days_since_first_quote']
        )
        result['process_engagement_intensity'].fillna(0, inplace=True)

    # Process switch indicator (changed preference recently)
    result['recent_process_switch'] = (
            (result['recent_process_adoption'] > 0.8) &
            (result['process_adoption_rate'] < 0.5)
    ).astype(int)

    # Process confidence score (combination of consistency and adoption)
    result['process_confidence_score'] = (
            result['process_consistency'] * 0.6 +
            np.clip(result['process_adoption_rate'], 0, 1) * 0.4
    )

    # --------------------------------------------------------------------
    # 6. FINAL REPORT
    # --------------------------------------------------------------------
    feature_count = len([col for col in result.columns if col != customer_col])
    print(f"\n✅ Created {feature_count} customer-level process features for {len(result):,} customers")

    print("\n📊 FEATURE SUMMARY:")
    print("-" * 50)

    summary_cols = ['process_adoption_rate', 'has_ever_used_process',
                    'process_consistency', 'process_preference']

    for col in summary_cols:
        if col in result.columns:
            if result[col].dtype in [np.int64, np.int32]:
                positive_pct = (result[col] > 0).mean() * 100
                print(f"{col:30} : {positive_pct:.1f}% positive")
            else:
                mean_val = result[col].mean()
                std_val = result[col].std()
                print(f"{col:30} : mean = {mean_val:.3f}, std = {std_val:.3f}")

    return result