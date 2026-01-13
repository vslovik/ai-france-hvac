import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_rf(X, y, model_name):
    print(f"  Data Frame Shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=50,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  AUC: {auc:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 5 features:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"    {i}. {row['feature']}: {row['importance']:.3f}")

    model_data = {
        'model': rf_model,
        'features': X.columns.tolist(),
        'feature_importance': feature_importance,
        'auc': auc,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Prediction model saved")
    print(f"✓ AUC: {auc:.3f}")
    print(f"✓ Features: {len(X.columns)}")

    return model_data