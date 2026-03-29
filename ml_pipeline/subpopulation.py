import pandas as pd

from ml_evaluation.error_analysis import comprehensive_error_analysis, analyze_feature_contribution_to_errors, \
    create_error_visualization
from ml_features.features import prepare_features
from ml_features.customer_features import create_customer_features
from ml_features.sequence_features  import create_sequence_features
from ml_features.brand_features import create_brand_features
from ml_features.model_features import create_model_features
from ml_features.market_features import create_market_features
from ml_features.equipment_features import create_equipment_features
from ml_features.solution_complexity_features import create_solution_complexity_features
from ml_features.timeline_features import create_timeline_features, create_advanced_timeline_features, create_timeline_interaction_features
from ml_features.role_features import create_commercial_role_features
from ml_features.process_features import create_process_features
from ml_features.correction_features import create_correction_features
from ml_features.catboost_interaction_features import create_catboost_interaction_features
from ml_features.efficiency_interation_features import create_efficiency_interaction_features
from ml_features.engagement_interation_features import create_engagement_interaction_features
from ml_features.advanced_features import create_advanced_interaction_features, create_conversion_pattern_features, create_precision_optimization_features, create_price_dominant_features
from ml_features.price_binning_features import create_price_binning_features
from ml_features.simulation_discovery import create_simulation_discovery_features
from ml_training.train_xgb import train_xgb
from ml_evaluation.dashboard import model_evaluation_report


def run_pipeline(df_quotes):
    df_quotes['dt_creation_devis'] = pd.to_datetime(df_quotes['dt_creation_devis'])
    print(f"\n📊 Original quote data: {len(df_quotes):,} quotes from {df_quotes['numero_compte'].nunique():,} customers")

    print("\n" + "=" * 80)
    print("🏗️  BUILDING FEATURES FOR SUBPOPULATION CUSTOMERS")
    print("=" * 80)

    feature_funcs = [create_customer_features, create_sequence_features, create_brand_features,
                     create_model_features, create_market_features,
                     create_equipment_features, create_solution_complexity_features,
                     create_timeline_features, create_advanced_timeline_features,
                     create_commercial_role_features, create_process_features,
                     create_correction_features
                     ]

    new_df = feature_funcs[0](df_quotes)
    for func in feature_funcs[1:]:
        new_df_ = func(df_quotes)
        new_df = pd.merge(new_df, new_df_, on='numero_compte', how='left', suffixes=('_dup', ''))
        new_df = new_df.drop(columns=[x for x in new_df.columns if '_dup' in x], errors='ignore')

    print(f"\n✅ Customer-level features created: {len(new_df):,} customers")

    print("\n" + "=" * 80)
    print("🔧 ADDING INTERACTION FEATURES")
    print("=" * 80)

    new_df = create_timeline_interaction_features(new_df)
    new_df, _ = create_catboost_interaction_features(new_df)
    new_df, _ = create_efficiency_interaction_features(new_df)
    new_df, _ = create_engagement_interaction_features(new_df)
    new_df = create_advanced_interaction_features(new_df)
    new_df = create_conversion_pattern_features(new_df)
    new_df = create_precision_optimization_features(new_df)
    new_df = create_price_dominant_features(new_df)
    new_df = create_price_binning_features(new_df)
    new_df = create_simulation_discovery_features(new_df)

    print("\n" + "=" * 80)
    print("🔧 ENCODING & PREPARING FOR MODELING")
    print("=" * 80)

    X = new_df.drop(columns=['numero_compte', 'converted'], errors='ignore')
    y = new_df['converted']
    X_clean, y_clean = prepare_features(X, y, "Cold Region Features")

    print(f"   Features: {X_clean.shape[1]}, Samples: {X_clean.shape[0]}")

    print("\n" + "=" * 80)
    print("🚀 TRAINING MODEL ON SUBPOPULATION CUSTOMERS")
    print("=" * 80)

    result = train_xgb(X_clean, y_clean, 'cold_region_model')

    print("\n" + "=" * 80)
    print("📊 MODEL EVALUATION - SUBPOPULATION CUSTOMERS")
    print("=" * 80)

    df_eval = result['X_test'].copy()
    df_eval['converted'] = result['y_test']

    model_evaluation_report(df_eval, result['model'], result['features'], 'converted')

    print("\n" + "=" * 80)
    print("🔍 FEATURE IMPORTANCE - SUBPOPULATION MODEL")
    print("=" * 80)

    feature_importance = pd.DataFrame({
        'feature': result['features'],
        'importance': result['model'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Features:")
    print(feature_importance.head(20))

    print("\n" + "=" * 80)
    print("🧪 SIMULATION-DISCOVERY FEATURES IN SUBPOPULATION MODEL")
    print("=" * 80)

    simulation_features = [
        'heat_pump_to_stove_opportunity',
        'boiler_to_ac_opportunity',
        'cold_region_heat_pump',
        'cold_heat_pump_to_stove',
        'follow_up_opportunity'
    ]

    print("\nFeature Importances for Simulation-Discovery Features:")
    for feat in simulation_features:
        if feat in feature_importance['feature'].values:
            imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
            rank = feature_importance[feature_importance['feature'] == feat].index[0] + 1
            print(f"  #{rank}: {feat}: {imp:.4f}")
        else:
            print(f"  ❌ {feat}: NOT FOUND")

    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING COMPLETE")
    print("=" * 80)

    return result


def run_model_error_analysis(X_test, y_test, model):
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 80)

    # Ensure X_test doesn't have the target column
    if 'converted' in X_test.columns:
        X_test = X_test.drop(columns=['converted'])

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Create a dataframe for error analysis that includes the predictions
    error_df = X_test.copy()
    error_df['converted'] = y_test
    error_df['predicted'] = y_pred
    if y_pred_proba is not None:
        error_df['prediction_probability'] = y_pred_proba

    error_results = comprehensive_error_analysis(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        original_df=error_df,  # Use the properly prepared dataframe
        customer_id_col='numero_compte'
    )

    analyze_feature_contribution_to_errors(model, X_test, y_test, y_pred)
    create_error_visualization(error_results, y_test, y_pred, y_pred_proba)
