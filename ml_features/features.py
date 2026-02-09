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