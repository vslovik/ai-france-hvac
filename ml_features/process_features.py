import pandas as pd
import numpy as np


def create_process_features(
        df: pd.DataFrame,
        quote_date_col: str = "dt_creation_devis",
        customer_col: str = "numero_compte",
        process_col: str = "fg_nouveau_process_relance_devis",
        target_col: str = "fg_devis_accepte",
) -> pd.DataFrame:
    """
    HYPER-OPTIMIZED: Removes all .apply(lambda) calls for maximum speed
    """
    required = {customer_col, process_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Error: Missing required columns: {missing}")
        return pd.DataFrame()

    df = df.copy()
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
    # 4. Historical success rates (if target exists)
    # ------------------------------------------------------------------------
    if target_col in df.columns:
        # Create columns for conditional success
        df["_success_new"] = np.where(df[process_col] == 1, df[target_col], np.nan)
        df["_success_old"] = np.where(df[process_col] == 0, df[target_col], np.nan)

        # Historical success rates WITHOUT .apply(lambda)
        df["hist_success_new_process"] = (
            df.groupby(customer_col)["_success_new"]
            .expanding()
            .mean()
            .groupby(level=0)
            .shift()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        df["hist_success_old_process"] = (
            df.groupby(customer_col)["_success_old"]
            .expanding()
            .mean()
            .groupby(level=0)
            .shift()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # ------------------------------------------------------------------------
    # 5. Select final columns
    # ------------------------------------------------------------------------
    feature_cols = [
        "current_process_new", "current_process_missing",
        "hist_total_quotes", "historical_process_adoption_rate",
        "historical_process_consistency", "historical_quotes_with_process",
        "is_first_quote", "process_consistency_with_history",
        "process_deviation_from_history",
    ]

    if target_col in df.columns:
        feature_cols.extend([
            "hist_success_new_process", "hist_success_old_process",
        ])

    result = df[[customer_col] + feature_cols].copy()

    print(f"Created {len(feature_cols)} process features")
    print(f"→ {len(result):,} quotes | {result[customer_col].nunique():,} customers")

    return result