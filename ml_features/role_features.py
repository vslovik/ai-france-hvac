import pandas as pd


def create_commercial_role_features(
        df: pd.DataFrame,
        commercial_col: str = "fonction_commercial",
        customer_col: str = "numero_compte"
) -> pd.DataFrame:
    """
    CONSERVATIVE SAFE VERSION: Only features that cannot leak
    """
    # Check required columns
    if commercial_col not in df.columns or customer_col not in df.columns:
        return pd.DataFrame(columns=[customer_col])

    df_work = df[[customer_col, commercial_col]].copy()

    # Group by customer
    g = df_work.groupby(customer_col)

    # ========== ONLY SAFE FEATURES ==========

    # 1. Basic flags (always safe)
    result = pd.DataFrame({
        'has_commercial_data': (g[commercial_col].count() > 0).astype(int),
    })

    # 2. Role consistency (safe - only compares within customer)
    result['commercial_role_consistency'] = (g[commercial_col].nunique() == 1).astype(int)

    # 3. Senior role detection (safe - based on role names)
    roles_upper = df_work[commercial_col].astype(str).str.upper()
    is_senior = roles_upper.str.contains('RESPONSABLE|DIRECTEUR|MANAGER|CHEF|SENIOR|EXPERT', na=False)
    df_work['is_senior'] = is_senior.astype(int)

    result['has_senior_commercial'] = g['is_senior'].max().astype(int)

    # ========== UNSAFE FEATURES - REMOVED ==========
    # ❌ total_quotes - leaks future quote count
    # ❌ unique_roles - leaks future role variety
    # ❌ senior_commercial_count - leaks future senior interactions

    result = result.reset_index()

    print(f"✅ Created {len(result.columns) - 1} LEAKAGE-SAFE commercial features")
    print("   REMOVED: total_quotes, unique_roles, senior_commercial_count (potential leakage)")

    return result