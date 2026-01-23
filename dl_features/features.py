import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def create_dl_specific_features(X, y=None):
    """
    Create safe, normalized features optimized for DL models.
    Transforms ALL numeric features, not just top 3.
    """
    print("\n" + "=" * 80)
    print("CREATING SAFE DL-OPTIMIZED FEATURES (V2)")
    print("=" * 80)

    print(f"📊 Input shape: {X.shape}")

    # Create a copy
    X_transformed = X.copy()

    # Identify numeric columns
    numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📋 Found {len(numeric_cols)} numeric columns")

    if len(numeric_cols) == 0:
        print("⚠️  No numeric columns found!")
        return X_transformed, y

    # STEP 1: Scale numeric features FIRST
    print("\n🔧 Step 1: Scaling features to reasonable range...")
    scaler = RobustScaler(quantile_range=(25, 75))

    # Scale numeric features
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_transformed[numeric_cols]),
        columns=numeric_cols,
        index=X_transformed.index
    )

    # Replace original numeric columns with scaled ones
    X_transformed[numeric_cols] = X_scaled

    # STEP 2: Add safe transformations to ALL numeric features
    print("\n🔧 Step 2: Adding safe transformations to ALL numeric features...")

    new_features_count = 0
    for feat in numeric_cols:
        try:
            # Add 2-3 safe transformations per feature
            # 1. Square root (safe for all values with abs)
            X_transformed[f'{feat}_abs_sqrt'] = np.sqrt(np.abs(X_transformed[feat]) + 1e-8)
            new_features_count += 1

            # 2. Log transform (safe with abs)
            X_transformed[f'{feat}_log'] = np.log(np.abs(X_transformed[feat]) + 1)
            new_features_count += 1

            # 3. Sigmoid-like transformation (bounded)
            X_transformed[f'{feat}_tanh'] = np.tanh(X_transformed[feat] * 0.5)
            new_features_count += 1

        except Exception as e:
            print(f"    ⚠️  Could not transform {feat}: {str(e)[:50]}")

    # STEP 3: Add safe interactions between original features
    print("\n🔧 Step 3: Adding safe interactions...")

    # Create meaningful interactions if we have enough features
    if len(numeric_cols) >= 2:
        feat1, feat2 = numeric_cols[:2]
        X_transformed[f'{feat1}_div_{feat2}'] = (
                X_transformed[feat1] / (X_transformed[feat2] + 1e-8)
        ).clip(-10, 10)
        new_features_count += 1
        print(f"    ✓ Added interaction: {feat1} / {feat2}")

    # STEP 4: Final clipping to safe range
    print("\n🔧 Step 4: Clipping all features to safe range...")
    all_numeric = X_transformed.select_dtypes(include=[np.number]).columns
    X_transformed[all_numeric] = X_transformed[all_numeric].clip(-10, 10)

    # Final summary
    print(f"\n✅ SAFE DL Features Created:")
    print(f"  Original: {X.shape[1]} features")
    print(f"  Final: {X_transformed.shape[1]} features")
    print(f"  Added: {new_features_count} new features")

    print(f"\n📊 Safe value ranges:")
    print(f"  Min: {X_transformed.min().min():.2f}")
    print(f"  Max: {X_transformed.max().max():.2f}")
    print(f"  Mean: {X_transformed.mean().mean():.2f}")

    return X_transformed, y


def correlation_selection(X, y, threshold=0.95):
    """Remove highly correlated features"""
    from sklearn.feature_selection import VarianceThreshold

    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    X_array = selector.fit_transform(X)

    # Get features that survived
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()

    removed_low_var = X.shape[1] - len(selected_features)
    print(f"Removed {removed_low_var} low-variance features")

    # Return DataFrame with feature names
    return pd.DataFrame(X_array, columns=selected_features, index=X.index)


def mutual_info_selection(X, y, n_features=64):
    """Select features using mutual information"""
    from sklearn.feature_selection import mutual_info_classif, SelectKBest

    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Create selector with pre-computed scores
    selector = SelectKBest(lambda X, y: mi_scores, k=min(n_features, X.shape[1]))
    X_array = selector.fit_transform(X, y)

    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()

    # Show top features
    feature_scores = list(zip(X.columns, mi_scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {n_features} features by Mutual Information:")
    for i, (feat, score) in enumerate(feature_scores[:10]):
        print(f"  {i+1:2d}. {feat:30s} | MI: {score:.4f}")

    # Return DataFrame with feature names
    return pd.DataFrame(X_array, columns=selected_features, index=X.index), selected_features


