import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


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