import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Quick check before running full function
def check_chronological_integrity(df, customer_col="numero_compte", date_col="dt_creation_devis"):
    """Quick check for chronological issues"""
    df_check = df[[customer_col, date_col]].copy()
    df_check[date_col] = pd.to_datetime(df_check[date_col], errors='coerce')

    issues = []
    for customer in df_check[customer_col].unique()[:100]:  # Check first 100 customers
        cust_dates = df_check[df_check[customer_col] == customer][date_col].dropna()
        if not cust_dates.is_monotonic_increasing:
            issues.append(customer)

    if issues:
        print(f"❌ Found {len(issues)} customers with non-chronological quotes")
        print(f"   Example customers: {issues[:5]}")
        return False
    else:
        print("✅ All checked customers have chronological quotes")
        return True


# Add this validation function to your pipeline
def validate_no_temporal_leakage(df, customer_col, date_col):
    """Ensure no data leakage from non-chronological data"""
    df_temp = df.sort_values([customer_col, date_col])

    # Create a sequence ID to check ordering
    df_temp['seq_id'] = df_temp.groupby(customer_col).cumcount()
    df_temp['orig_index'] = df_temp.index

    # Check if original order matches chronological order
    df_check = df_temp.groupby(customer_col).apply(
        lambda x: not x.index.equals(x.sort_values(date_col).index)
    )

    problematic_customers = df_check[df_check].index.tolist()

    if problematic_customers:
        print(f"🚨 DATA LEAKAGE RISK: {len(problematic_customers)} customers need re-sorting")
        return False, problematic_customers
    return True, []


def prepare_dataset_without_leakage(df, dataset_name):
    """
    Prepare dataset WITHOUT data leakage
    Remove features that contain future information
    """
    print(f"\n📊 Re-preparing {dataset_name} without leakage...")

    df_model = df.copy()
    target = 'converted'
    y = df_model[target]

    # REMOVE LEAKY FEATURES
    leaky_features = [
        'customer_total_sales',  # Contains future conversion info!
        'customer_avg_price',  # Based on all quotes (including future)
        'customer_product_variety',  # Based on all products (including future)
        'products_considered',  # Full list including future quotes
        'product_types'  # Full list including future quotes
    ]

    # Also remove temporal features that shouldn't be known upfront
    exclude_features = ['numero_compte', 'customer_opportunity_id',
                        'customer_session_id', 'opportunity_id', 'session_id',
                        'start_date', 'end_date', 'first_quote_date', 'last_quote_date',
                        'main_product_family'] + leaky_features

    available_exclude = [f for f in exclude_features if f in df_model.columns]
    X = df_model.drop(columns=available_exclude + [target], errors='ignore')

    print(f"  Features after removing leakage: {X.shape[1]}")
    print(f"  Removed leaky features: {[f for f in leaky_features if f in df_model.columns]}")

    # Handle missing values and encode
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna('missing')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y