import pandas as pd


def create_commercial_role_features(
        df: pd.DataFrame,
        commercial_col: str = "fonction_commercial",
        customer_col: str = "numero_compte",
        date_col: str = "dt_creation_devis",
        first_purchase_dates: dict = None
) -> pd.DataFrame:
    """
    CONSERVATIVE SAFE VERSION: Only features that cannot leak
    Now includes FIRST CONVERSION filtering
    """
    # Check required columns
    if commercial_col not in df.columns or customer_col not in df.columns:
        return pd.DataFrame(columns=[customer_col])

    df_work = df[[customer_col, commercial_col]].copy()

    # ========== FIRST CONVERSION FILTERING ==========
    print("🔍 Applying first conversion filtering...")
    if first_purchase_dates is not None and date_col in df.columns:
        df_work[date_col] = df[date_col]
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors="coerce")

        pre_filter_count = len(df_work)

        # Vectorized filtering
        if date_col in df_work.columns:
            # Create series of first purchase dates for each customer
            df_work['first_purchase_date'] = df_work[customer_col].map(first_purchase_dates)

            # Keep rows where:
            # 1. No first purchase date (never converters) OR
            # 2. Quote date <= first purchase date
            mask = df_work['first_purchase_date'].isna() | (df_work[date_col] <= df_work['first_purchase_date'])
            df_work = df_work[mask].reset_index(drop=True)

            # Drop temporary column
            df_work = df_work.drop(columns=[date_col, 'first_purchase_date'])
        else:
            # If no date column, we can't filter chronologically
            print("⚠️  No date column - cannot filter by first purchase dates")

        post_filter_count = len(df_work)
        print(f"   Filtered: {pre_filter_count:,} → {post_filter_count:,} quotes")
        print(f"   Removed {pre_filter_count - post_filter_count:,} post-first-purchase quotes")
    else:
        print("⚠️  No first_purchase_dates provided - using all data")

    # ========== ONLY SAFE FEATURES ==========

    # 1. Basic flags (always safe)
    result = pd.DataFrame({
        'has_commercial_data': (df_work.groupby(customer_col)[commercial_col].count() > 0).astype(int),
    })

    # 2. Role consistency (safe - only compares within customer)
    result['commercial_role_consistency'] = (df_work.groupby(customer_col)[commercial_col].nunique() == 1).astype(int)

    # 3. Senior role detection (safe - based on role names)
    roles_upper = df_work[commercial_col].astype(str).str.upper()
    is_senior = roles_upper.str.contains('RESPONSABLE|DIRECTEUR|MANAGER|CHEF|SENIOR|EXPERT', na=False)
    df_work['is_senior'] = is_senior.astype(int)

    result['has_senior_commercial'] = df_work.groupby(customer_col)['is_senior'].max().astype(int)

    # ========== UNSAFE FEATURES - REMOVED ==========
    # ❌ total_quotes - leaks future quote count
    # ❌ unique_roles - leaks future role variety
    # ❌ senior_commercial_count - leaks future senior interactions

    result = result.reset_index()

    print(f"✅ Created {len(result.columns) - 1} LEAKAGE-SAFE commercial features")
    print("   REMOVED: total_quotes, unique_roles, senior_commercial_count (potential leakage)")
    print(f"   FIRST CONVERSION MODE: {'ENABLED' if first_purchase_dates is not None else 'DISABLED'}")

    return result