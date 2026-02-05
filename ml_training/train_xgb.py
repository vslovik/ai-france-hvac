import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import wandb


def train_xgb(X, y, model_name, wandb_run=None):
    """
    Train XGBoost classifier with optional wandb tracking

    Args:
        X: Features
        y: Target
        model_name: Name for saving model
        wandb_run: Optional wandb run object for tracking
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Default hyperparameters (based on your roadmap)
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        scale_pos_weight=len(y[y == 0]) / len(y[y == 1]) if sum(y) > 0 else 1  # Balanced class weighting
    )

    xgb_model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'conversion_rate': y.mean()
    }

    # ============ WANDB LOGGING ============
    if wandb_run:
        # Log metrics
        wandb_run.log({
            f"{model_name}/auc": metrics['auc'],
            f"{model_name}/accuracy": metrics['accuracy'],
            f"{model_name}/f1": metrics['f1'],
            f"{model_name}/train_size": metrics['train_size'],
            f"{model_name}/test_size": metrics['test_size'],
            f"{model_name}/conversion_rate": metrics['conversion_rate'],
            f"{model_name}/n_features": X.shape[1]
        })

        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Log top features
        wandb_run.summary[f"{model_name}_top_feature"] = feature_importance.iloc[0]['feature']
        wandb_run.summary[f"{model_name}_top_importance"] = feature_importance.iloc[0]['importance']

        # Log feature importance table
        importance_table = wandb.Table(dataframe=feature_importance.head(20))
        wandb_run.log({f"{model_name}/feature_importance": importance_table})

    # Save model
    model_data = {
        'model': xgb_model,
        'features': X.columns.tolist(),
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test
    }

    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved: {model_name}.pkl")
    print(f"✓ AUC: {metrics['auc']:.3f}")
    print(f"✓ F1 Score: {metrics['f1']:.3f}")
    print(f"✓ Training samples: {metrics['train_size']}")
    print(f"✓ Test samples: {metrics['test_size']}")

    return model_data