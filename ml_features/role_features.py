import pandas as pd
import re


def create_commercial_role_features(
        df: pd.DataFrame,
        commercial_col: str = "fonction_createur",  # Updated column name
        customer_col: str = "numero_compte",
        date_col: str = "dt_creation_devis",
        first_purchase_dates: dict = None
) -> pd.DataFrame:
    """
    ENHANCED SAFE VERSION: Adds more features while preserving vectorization
    Now includes the new fonction_createur column
    """
    # Check required columns
    if commercial_col not in df.columns or customer_col not in df.columns:
        return pd.DataFrame(columns=[customer_col])

    df_work = df[[customer_col, commercial_col]].copy()
    if date_col in df.columns:
        df_work[date_col] = df[date_col]

    # ========== FIRST CONVERSION FILTERING ==========
    print("🔍 Applying first conversion filtering...")
    if first_purchase_dates is not None and date_col in df.columns:
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors="coerce")

        pre_filter_count = len(df_work)

        # Vectorized filtering
        df_work['first_purchase_date'] = df_work[customer_col].map(first_purchase_dates)

        # Keep rows where:
        # 1. No first purchase date (never converters) OR
        # 2. Quote date <= first purchase date
        mask = df_work['first_purchase_date'].isna() | (df_work[date_col] <= df_work['first_purchase_date'])
        df_work = df_work[mask].reset_index(drop=True)

        # Drop temporary column
        df_work = df_work.drop(columns=[date_col, 'first_purchase_date'])

        post_filter_count = len(df_work)
        print(f"   Filtered: {pre_filter_count:,} → {post_filter_count:,} quotes")
    else:
        print("⚠️  No first_purchase_dates provided - using all data")

    # ========== ENHANCED VECTORIZED FEATURES ==========
    print("⚡ Creating enhanced role features...")

    # Pre-compute role groups for vectorized operations
    role_mapping = {
        'Commercial': 'Commercial',
        'Technico-Commercial': 'Technical',
        'Technico Commercial': 'Technical',
        'Technicien': 'Technical',
        'Technicien chauffagiste': 'Technical',
        'Assistante': 'Admin',
        'Assistante Administrative': 'Admin',
        'Admin SAV': 'Admin',
        'Responsable': 'Management',
        'Responsable exploitation': 'Management',
        'Directeur': 'Management',
    }

    # Vectorized role grouping
    df_work['role_group'] = df_work[commercial_col].map(role_mapping).fillna('Other')

    # Group by customer
    grouped = df_work.groupby(customer_col)

    # 1. Basic flags (always safe)
    result = pd.DataFrame({
        'has_commercial_data': (grouped[commercial_col].count() > 0).astype(int),
    })

    # 2. Role consistency (safe - only compares within customer)
    result['commercial_role_consistency'] = (grouped['role_group'].nunique() == 1).astype(int)

    # 3. Role diversity (how many different roles this customer interacted with)
    result['commercial_role_diversity'] = grouped['role_group'].nunique()

    # 4. Senior role detection (vectorized)
    senior_pattern = re.compile(r'RESPONSABLE|DIRECTEUR|MANAGER|CHEF|SENIOR|EXPERT', re.IGNORECASE)
    df_work['is_senior'] = df_work[commercial_col].fillna('').str.upper().str.contains(senior_pattern).astype(int)
    result['has_senior_commercial'] = grouped['is_senior'].max().astype(int)

    # 5. Primary role type (most frequent)
    def get_primary_role(roles):
        if len(roles) == 0:
            return 'unknown'
        return roles.value_counts().index[0]

    result['primary_role_type'] = grouped['role_group'].apply(get_primary_role)

    # 6. Role stability (ratio of most common role to total)
    def get_role_stability(roles):
        if len(roles) == 0:
            return 0
        return roles.value_counts().iloc[0] / len(roles)

    result['role_stability'] = grouped['role_group'].apply(get_role_stability)

    # 7. Number of commercial interactions
    result['commercial_interaction_count'] = grouped[commercial_col].count()

    # Reset index
    result = result.reset_index()
    result = result.rename(columns={customer_col: 'numero_compte'})

    print(f"✅ Created {len(result.columns) - 1} LEAKAGE-SAFE commercial features")
    print(f"   New features: role_diversity, primary_role_type, role_stability, interaction_count")

    return result