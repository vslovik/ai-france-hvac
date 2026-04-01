import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ml_features.brand_features import create_brand_features
from ml_features.catboost_interaction_features import create_catboost_interaction_features
from ml_features.correction_features import create_correction_features
from ml_features.customer_features import create_customer_features
from ml_features.efficiency_interation_features import create_efficiency_interaction_features
from ml_features.engagement_interation_features import create_engagement_interaction_features
from ml_features.equipment_features import create_equipment_features
from ml_features.market_features import create_market_features
from ml_features.model_features import create_model_features
from ml_features.process_features import create_process_features
from ml_features.role_features import create_commercial_role_features
from ml_features.sequence_features import create_sequence_features
from ml_features.solution_complexity_features import create_solution_complexity_features
from ml_features.timeline_features import create_timeline_features, create_advanced_timeline_features, \
    create_timeline_interaction_features


def create_features(df_quotes, dataset_name="New Features"):
    print("\n" + "=" * 80)
    print("STRATEGY: CREATE FEATURES")
    print("=" * 80)

    feature_funcs = [create_customer_features, create_sequence_features, create_brand_features,
                     create_model_features, create_market_features,
                     create_equipment_features, create_solution_complexity_features,
                     create_timeline_features, create_advanced_timeline_features,
                     create_commercial_role_features, create_process_features, create_correction_features]

    new_df = feature_funcs[0](df_quotes)
    for func in feature_funcs[1:]:
        new_df_ = func(df_quotes)
        new_df = pd.merge(new_df, new_df_, on='numero_compte', how='left', suffixes=('_dup', ''))
        new_df = new_df.drop(columns=[x for x in new_df.columns if '_dup' in x], errors='ignore')
        print(len(new_df))
        if func == create_sequence_features: sequence_df = new_df

    # Now it's clear which column is which
    y_new = new_df['converted']  # From sequence features
    new_df = create_timeline_interaction_features(new_df)
    new_df, _ = create_catboost_interaction_features(new_df)
    new_df, _ = create_efficiency_interaction_features(new_df)
    new_df, _ = create_engagement_interaction_features(new_df)

    X_new = new_df
             #.drop(columns=['numero_compte', 'converted'], errors='ignore')
    X_new_clean, y_new_clean = prepare_features(X_new, y_new, dataset_name)
    df_features = X_new_clean.copy()
    df_features['converted'] = y_new_clean
    print(f"\n✅ FEATURES CREATED: {df_features.shape[1]} features for {len(df_features)} samples")
    output_file = 'customer_features.csv'
    df_features.to_csv(output_file, index=False)
    print(f"  Features saved to {output_file}")
    return df_features


def prepare_features(X, y, dataset_name, encoders=None, feature_order=None):
    """
    Prepare feature matrix for modeling with consistent feature order.

    Parameters:
    - X: Features DataFrame
    - y: Target Series (or None for test set)
    - dataset_name: Name for logging
    - encoders: Dict of fitted LabelEncoders (for test set only)
    - feature_order: List of column names in fixed order (for test set)

    Returns:
    - X_clean: Prepared features (columns in fixed order)
    - y: Target (unchanged)
    - encoders: Dict of fitted LabelEncoders (for training set)
    - feature_order: List of column names in fixed order (for training set)
    """
    print(f"\n🔧 ENCODING & PREPARING FOR MODELING...")
    print(f"  Preparing {dataset_name}...")

    X_clean = X.copy()

    # Handle categorical columns
    categorical_cols = X_clean.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'numero_compte']

    # If encoders provided (test set), use them
    if encoders is not None:
        for col in categorical_cols:
            if col in encoders:
                X_clean[col] = X_clean[col].fillna('missing')
                # Replace unseen categories with 'missing'
                known_categories = set(encoders[col].classes_)
                X_clean[col] = X_clean[col].astype(str).apply(
                    lambda x: x if x in known_categories else 'missing'
                )
                # Ensure 'missing' is in the encoder's classes
                if 'missing' not in known_categories:
                    current_classes = encoders[col].classes_.tolist()
                    if 'missing' not in current_classes:
                        current_classes.append('missing')
                        new_encoder = LabelEncoder()
                        new_encoder.fit(current_classes)
                        X_clean[col] = new_encoder.transform(X_clean[col])
                        encoders[col] = new_encoder
                    else:
                        X_clean[col] = encoders[col].transform(X_clean[col])
                else:
                    X_clean[col] = encoders[col].transform(X_clean[col])

        # Enforce feature order if provided
        if feature_order is not None:
            # Ensure all columns exist
            missing_cols = set(feature_order) - set(X_clean.columns)
            if missing_cols:
                for col in missing_cols:
                    X_clean[col] = 0
            # Reorder columns
            X_clean = X_clean[feature_order]

        return X_clean, y
    else:
        # Training set: fit encoders
        encoders = {}
        for col in categorical_cols:
            X_clean[col] = X_clean[col].fillna('missing')
            # Add 'missing' to categories
            all_categories = X_clean[col].astype(str).unique().tolist()
            if 'missing' not in all_categories:
                all_categories.append('missing')
            le = LabelEncoder()
            le.fit(all_categories)
            X_clean[col] = le.transform(X_clean[col].astype(str))
            encoders[col] = le

        # Get feature order (sorted for consistency)
        feature_order = sorted(X_clean.columns.tolist())
        X_clean = X_clean[feature_order]

        return X_clean, y, encoders, feature_order