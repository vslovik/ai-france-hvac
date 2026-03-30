import time
import numpy as np
import pandas as pd

WINDOW_DAYS = 180


def create_sequence_features(
    df: pd.DataFrame,
    target_type: str = 'first_conversion',
    window_days: int = 30,
    date_col: str = 'dt_creation_devis',
    customer_col: str = 'numero_compte',
    accept_col: str = 'fg_devis_accepte',
    price_col: str = 'mt_apres_remise_ht_devis',
    product_col: str = 'regroup_famille_equipement_produit_principal',
    agency_col: str = 'nom_agence',
    region_col: str = 'nom_region',
    discount_col: str = 'mt_remise_exceptionnelle_ht',
    ttc_col: str = 'mt_ttc_apres_aide_devis'
) -> pd.DataFrame:
    """
    Create leakage-free sequence features for predicting first conversion.
    """
    print("=" * 80)
    print(f"CREATING SEQUENCE FEATURES (mode: {target_type})")
    print("=" * 80)
    start_time = time.time()

    # Data preparation
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values([customer_col, date_col]).reset_index(drop=True)

    print(f"  Total quotes: {len(df):,}")
    print(f"  Total customers: {df[customer_col].nunique():,}")
    print(f"  Using product column: {product_col}")
    print(f"  Window days: {window_days}")

    # Filter for first_conversion mode
    if target_type == 'first_conversion':
        print("  Filtering post-first-purchase data...")
        df['_conv_cumsum'] = df.groupby(customer_col)[accept_col].cumsum()
        df_filtered = df[df['_conv_cumsum'] <= 1].copy()
        df_filtered = df_filtered.drop(columns=['_conv_cumsum'])
        print(f"    Customers after filter: {df_filtered[customer_col].nunique():,}")
        print(f"    Quotes after filter: {len(df_filtered):,}")
    else:
        df_filtered = df.copy()

    # Store original for target calculation
    df_original = df.copy()

    # Fast vectorized preprocessing
    df_filtered['_days'] = (df_filtered.groupby(customer_col)[date_col].diff().dt.days).fillna(0)
    df_filtered['_product'] = df_filtered[product_col].fillna('unknown').astype(str)

    # Group indices for efficient processing
    customer_groups = df_filtered.groupby(customer_col).indices
    customer_ids = list(customer_groups.keys())
    n_customers = len(customer_ids)

    # Convert to arrays for speed
    dates_array = pd.to_datetime(df_filtered[date_col]).values.astype('datetime64[ns]')
    prices_array = df_filtered[price_col].values.astype(float)
    converted_array = df_filtered[accept_col].values.astype(bool)
    products_array = df_filtered['_product'].values

    # Pre-allocate result arrays
    result_arrays = {
        customer_col: customer_ids,
        'converted': np.zeros(n_customers, dtype=int),
        'total_historical_quotes': np.zeros(n_customers, dtype=int),
        'had_historical_quotes': np.zeros(n_customers, dtype=int),
        'avg_days_since_first_quote': np.zeros(n_customers, dtype=float),
        'std_days_since_first_quote': np.zeros(n_customers, dtype=float),
        'avg_recent_quote_count': np.zeros(n_customers, dtype=float),
        'std_recent_quote_count': np.zeros(n_customers, dtype=float),
        'avg_recent_avg_price': np.zeros(n_customers, dtype=float),
        'std_recent_avg_price': np.zeros(n_customers, dtype=float),
        'avg_recent_price_std': np.zeros(n_customers, dtype=float),
        'std_recent_price_std': np.zeros(n_customers, dtype=float),
        'avg_recent_product_variety': np.zeros(n_customers, dtype=float),
        'std_recent_product_variety': np.zeros(n_customers, dtype=float),
        'avg_current_price': np.zeros(n_customers, dtype=float),
        'std_current_price': np.zeros(n_customers, dtype=float),
        'price_trend': np.zeros(n_customers, dtype=float)
    }

    print(f"⚡ Processing {n_customers:,} customers with sequence features...")

    # Process each customer
    for i, customer_id in enumerate(customer_ids):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i:,}/{n_customers:,} customers")

        indices = customer_groups[customer_id]
        n_quotes = len(indices)

        # Get local arrays for this customer (relative indices within customer)
        cust_dates = dates_array[indices]
        cust_prices = prices_array[indices]
        cust_converted = converted_array[indices]
        cust_products = products_array[indices]

        # Find first conversion (within this customer's quotes)
        conv_mask = cust_converted
        conv_indices = np.where(conv_mask)[0]
        converted = 1 if len(conv_indices) > 0 else 0
        result_arrays['converted'][i] = converted

        # Determine historical data (before prediction point)
        if converted:
            # Converters: use quotes BEFORE first conversion
            first_conv_idx = conv_indices[0]
            # Historical quotes are those BEFORE the first conversion (index 0 to first_conv_idx-1)
            # We need the local indices within this customer's quotes
            hist_local_indices = list(range(first_conv_idx))  # Local indices 0 to first_conv_idx-1
            historical_indices = indices[hist_local_indices] if len(hist_local_indices) > 0 else []
        else:
            # Non-converters: use all quotes except last (predict at last quote)
            hist_local_indices = list(range(n_quotes - 1)) if n_quotes > 1 else []
            historical_indices = indices[hist_local_indices] if len(hist_local_indices) > 0 else []

        n_hist = len(historical_indices)
        result_arrays['total_historical_quotes'][i] = n_hist
        result_arrays['had_historical_quotes'][i] = 1 if n_hist > 0 else 0

        # Get local historical data for this customer
        if n_hist == 0:
            # Use the first quote price as baseline if available
            if n_quotes > 0:
                result_arrays['avg_current_price'][i] = cust_prices[0]
            continue

        if n_hist == 1:
            # Use the single historical quote price
            result_arrays['avg_current_price'][i] = cust_prices[hist_local_indices[0]]
            continue

        # Get historical data arrays (using local indices)
        hist_dates = cust_dates[hist_local_indices]
        hist_prices = cust_prices[hist_local_indices]
        hist_products = cust_products[hist_local_indices]

        # Calculate days since first historical quote
        days_since_first = (hist_dates - hist_dates[0]).astype('timedelta64[D]').astype(int)

        # Prepare sequence data
        seq_days = []
        seq_counts = []
        seq_avg_prices = []
        seq_price_stds = []
        seq_product_varieties = []
        seq_current_prices = []

        # For each historical quote (except first), calculate window features
        for j in range(1, n_hist):
            current_date = hist_dates[j]
            window_start = current_date - np.timedelta64(window_days, 'D')

            # Find quotes in window BEFORE current quote
            mask = (hist_dates[:j] >= window_start)
            recent_idx = np.where(mask)[0]

            if len(recent_idx) > 0:
                recent_prices = hist_prices[recent_idx]
                unique_products = len(set(hist_products[recent_idx]))

                seq_days.append(days_since_first[j])
                seq_counts.append(len(recent_idx))
                seq_avg_prices.append(np.mean(recent_prices))
                seq_price_stds.append(np.std(recent_prices) if len(recent_idx) > 1 else 0)
                seq_product_varieties.append(unique_products)
                seq_current_prices.append(hist_prices[j])

        if not seq_days:
            result_arrays['avg_current_price'][i] = hist_prices[-1]
            continue

        # Convert to arrays and calculate statistics
        seq_days_arr = np.array(seq_days)
        seq_counts_arr = np.array(seq_counts)
        seq_avg_prices_arr = np.array(seq_avg_prices)
        seq_price_stds_arr = np.array(seq_price_stds)
        seq_product_varieties_arr = np.array(seq_product_varieties)
        seq_current_prices_arr = np.array(seq_current_prices)

        result_arrays['avg_days_since_first_quote'][i] = np.mean(seq_days_arr)
        result_arrays['avg_recent_quote_count'][i] = np.mean(seq_counts_arr)
        result_arrays['avg_recent_avg_price'][i] = np.mean(seq_avg_prices_arr)
        result_arrays['avg_recent_price_std'][i] = np.mean(seq_price_stds_arr)
        result_arrays['avg_recent_product_variety'][i] = np.mean(seq_product_varieties_arr)
        result_arrays['avg_current_price'][i] = np.mean(seq_current_prices_arr)

        if len(seq_days_arr) > 1:
            result_arrays['std_days_since_first_quote'][i] = np.std(seq_days_arr)
            result_arrays['std_recent_quote_count'][i] = np.std(seq_counts_arr)
            result_arrays['std_recent_avg_price'][i] = np.std(seq_avg_prices_arr)
            result_arrays['std_recent_price_std'][i] = np.std(seq_price_stds_arr)
            result_arrays['std_recent_product_variety'][i] = np.std(seq_product_varieties_arr)
            result_arrays['std_current_price'][i] = np.std(seq_current_prices_arr)

        if len(seq_current_prices_arr) > 1:
            x = np.arange(len(seq_current_prices_arr))
            result_arrays['price_trend'][i] = np.polyfit(x, seq_current_prices_arr, 1)[0]

    # Create DataFrame with results
    df_result = pd.DataFrame(result_arrays)

    # Add categorical features from original data
    print("\n  Adding categorical features...")

    categorical_cols = [agency_col, region_col]
    for col in categorical_cols:
        if col in df.columns:
            mode_df = df.groupby(customer_col)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing')
            df_result = df_result.merge(mode_df.rename(col), left_on=customer_col, right_index=True, how='left')

    # Add derived features
    df_result['price_range'] = df_result['avg_current_price'] - df_result['avg_recent_avg_price']
    df_result['product_consistency'] = (df_result['avg_recent_product_variety'] <= 1).astype(int)
    df_result['engagement_density'] = df_result['avg_recent_quote_count'] / window_days

    # Fill NaN values
    fill_cols = ['price_range', 'product_consistency', 'engagement_density',
                 'avg_days_since_first_quote', 'std_days_since_first_quote',
                 'avg_recent_quote_count', 'std_recent_quote_count',
                 'avg_recent_avg_price', 'std_recent_avg_price',
                 'avg_recent_price_std', 'std_recent_price_std',
                 'avg_recent_product_variety', 'std_recent_product_variety',
                 'avg_current_price', 'std_current_price', 'price_trend']

    for col in fill_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna(0)

    # Final column selection
    final_columns = [
        customer_col,
        'converted',
        'total_historical_quotes',
        'had_historical_quotes',
        'engagement_density',
        'avg_days_since_first_quote',
        'std_days_since_first_quote',
        'avg_recent_quote_count',
        'std_recent_quote_count',
        'avg_recent_avg_price',
        'std_recent_avg_price',
        'avg_recent_price_std',
        'std_recent_price_std',
        'avg_recent_product_variety',
        'std_recent_product_variety',
        'avg_current_price',
        'std_current_price',
        'price_range',
        'price_trend',
        'product_consistency',
        agency_col,
        region_col
    ]

    df_result = df_result[[col for col in final_columns if col in df_result.columns]]

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ SEQUENCE FEATURES CREATED")
    print("=" * 80)
    print(f"  Total customers: {len(df_result):,}")
    print(f"  Converters: {df_result['converted'].sum():,} ({df_result['converted'].mean() * 100:.1f}%)")
    print(f"  Features created: {len(df_result.columns) - 1}")
    print(f"  Product column used: {product_col}")
    print(f"  Execution time: {elapsed:.1f} seconds")

    return df_result