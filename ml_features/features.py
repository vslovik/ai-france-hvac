import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def prepare_features(X, y, dataset_name):
    print("\n🔧 ENCODING & PREPARING FOR MODELING...")
    """Prepare feature matrix for modeling"""
    print(f"  Preparing {dataset_name}...")

    X_clean = X.copy()

    # Handle categorical
    categorical_cols = X_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X_clean[col] = X_clean[col].fillna('missing')
        le = LabelEncoder()
        X_clean[col] = le.fit_transform(X_clean[col].astype(str))

    # Handle missing values
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X_clean[col] = X_clean[col].fillna(X_clean[col].median())

    print(f"  Features: {X_clean.shape[1]}, Samples: {len(X_clean)}")
    return X_clean, y