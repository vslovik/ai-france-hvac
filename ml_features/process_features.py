import pandas as pd
import numpy as np


def create_process_features(
        df: pd.DataFrame,
        quote_date_col: str = "dt_creation_devis",
        customer_col: str = "numero_compte",
        process_col: str = "fg_nouveau_process_relance_devis",
        target_col: str = "fg_devis_accepte",
        first_purchase_dates: dict = None
) -> pd.DataFrame:
    """
    HYPER-OPTIMIZED: Removes all .apply(lambda) calls for maximum speed
    Now includes FIRST CONVERSION filtering
    """
    required = {customer_col, process_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Error: Missing required columns: {missing}")
        return pd.DataFrame()

    df = df.copy()

    # ========== FIRST CONVERSION FILTERING ==========
    print("🔍 Applying first conversion filtering...")
    if first_purchase_dates is not None and quote_date_col in df.columns:
        df[quote_date_col] = pd.to_datetime(df[quote_date_col], errors="coerce")

        pre_filter_count = len(df)

        # Vectorized filtering
        if quote_date_col in df.columns:
            # Create series of first purchase dates for each customer
            df['first_purchase_date'] = df[customer_col].map(first_purchase_dates)

            # Keep rows where:
            # 1. No first purchase date (never converters) OR
            # 2. Quote date <= first purchase date
            mask = df['first_purchase_date'].isna() | (df[quote_date_col] <= df['first_purchase_date'])
            df = df[mask].reset_index(drop=True)

            # Drop temporary column
            df = df.drop(columns=['first_purchase_date'])
        else:
            # If no date column, we can't filter chronologically
            print("⚠️  No date column - cannot filter by first purchase dates")

        post_filter_count = len(df)
        print(f"   Filtered: {pre_filter_count:,} → {post_filter_count:,} quotes")
        print(f"   Removed {pre_filter_count - post_filter_count:,} post-first-purchase quotes")
    else:
        print("⚠️  No first_purchase_dates provided - using all data")

    if quote_date_col in df.columns:
        df[quote_date_col] = pd.to_datetime(df[quote_date_col], errors="coerce")

    # Sort once
    sort_keys = [customer_col]
    if quote_date_col in df.columns:
        sort_keys.append(quote_date_col)
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # ------------------------------------------------------------------------
    # 1. Basic features
    # ------------------------------------------------------------------------
    df["quote_seq"] = df.groupby(customer_col).cumcount()
    df["hist_total_quotes"] = df["quote_seq"]
    df["current_process_new"] = (df[process_col] == 1).astype(int)
    df["current_process_missing"] = df[process_col].isna().astype(int)
    df["is_first_quote"] = (df["quote_seq"] == 0).astype(int)

    # Create numeric process column
    df["_process_numeric"] = (df[process_col] == 1).astype(float)
    df.loc[df[process_col].isna(), "_process_numeric"] = np.nan

    # ------------------------------------------------------------------------
    # 2. Historical features WITHOUT .apply(lambda)
    # ------------------------------------------------------------------------

    # Historical adoption rate
    df["historical_process_adoption_rate"] = (
        df.groupby(customer_col)["_process_numeric"]
        .expanding()
        .mean()
        .groupby(level=0)
        .shift()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Historical consistency (std == 0)
    expanding_std = (
        df.groupby(customer_col)["_process_numeric"]
        .expanding()
        .std()
        .groupby(level=0)
        .shift()
        .reset_index(level=0, drop=True)
    )
    df["historical_process_consistency"] = ((expanding_std == 0) | expanding_std.isna()).astype(int)

    # Historical quotes with process
    df["historical_quotes_with_process"] = (
        df.groupby(customer_col)["_process_numeric"]
        .expanding()
        .count()
        .groupby(level=0)
        .shift()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # 3. Derived features
    # ------------------------------------------------------------------------
    df["process_consistency_with_history"] = np.where(
        df["quote_seq"] == 0,
        1,
        (df["current_process_new"] == df["historical_process_adoption_rate"].round()).astype(int)
    )

    df["process_deviation_from_history"] = np.abs(
        df["current_process_new"] - df["historical_process_adoption_rate"]
    )

    # ------------------------------------------------------------------------
    # 4. Select final columns
    # ------------------------------------------------------------------------
    feature_cols = [
        "current_process_new", "current_process_missing",
        "hist_total_quotes", "historical_process_adoption_rate",
        "historical_process_consistency", "historical_quotes_with_process",
        "is_first_quote", "process_consistency_with_history",
        "process_deviation_from_history",
    ]

    result = df[[customer_col] + feature_cols].copy()

    print(f"Created {len(feature_cols)} process features")
    print(f"→ {len(result):,} quotes | {result[customer_col].nunique():,} customers")
    print(f"→ FIRST CONVERSION MODE: {'ENABLED' if first_purchase_dates is not None else 'DISABLED'}")

    return result