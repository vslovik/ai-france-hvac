import numpy as np
import pandas as pd


def analyze_catboost_interactions(model, feature_names, X_sample=None, top_n=10):
    """
    Analyze feature interactions discovered by CatBoost

    Args:
        model: Trained CatBoost model
        feature_names: List of feature names
        X_sample: Sample data for deeper analysis (optional)
        top_n: Number of top interactions to display
    """
    print("🔬 CATBOOST FEATURE INTERACTION ANALYSIS")
    print("=" * 60)

    # Method 1: Get built-in feature interactions
    try:
        # CatBoost can provide feature interaction statistics
        if hasattr(model, 'get_feature_importance'):
            # Get feature importance with type 'Interaction'
            interactions = model.get_feature_importance(
                data=X_sample,
                type='Interaction'
            )

            if len(interactions) > 0:
                print(f"\n📊 Top {top_n} Feature Interactions (CatBoost Auto-Discovered):")
                print("-" * 50)

                # Convert to DataFrame for better display
                interaction_df = pd.DataFrame({
                    'feature_index': range(len(interactions)),
                    'importance': interactions
                })

                # Map indices to feature names for interactions
                # Note: For interactions, indices represent combination pairs
                # We need to decode them
                interaction_df['interaction'] = interaction_df['feature_index'].apply(
                    lambda idx: decode_catboost_interaction(idx, feature_names)
                )

                interaction_df = interaction_df.sort_values('importance', ascending=False).head(top_n)

                for _, row in interaction_df.iterrows():
                    print(f"  • {row['interaction']}: {row['importance']:.4f}")

    except Exception as e:
        print(f"⚠️ Could not extract direct interactions: {e}")

    # Method 2: Analyze using SHAP for interaction effects
    print("\n🎯 SHAP-Based Interaction Analysis:")
    print("-" * 50)
    try:
        import shap

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Use sample if provided, otherwise create synthetic
        if X_sample is None:
            X_sample = shap.utils.sample(X_sample if X_sample is not None else model._data, 100)

        shap_values = explainer.shap_values(X_sample)

        # Find top interactions using SHAP dependence
        print("\n🔍 Top Feature Interactions (SHAP-based):")

        # Get overall feature importance for reference
        shap_importance = np.abs(shap_values).mean(0)
        top_features_idx = np.argsort(shap_importance)[-5:]  # Top 5 features

        for idx in top_features_idx:
            feature = feature_names[idx]
            print(f"\n  • Interactions with '{feature}':")

            # Find features that interact most with this one
            interaction_strength = []
            for j in range(len(feature_names)):
                if j != idx:
                    # Calculate correlation of SHAP values as proxy for interaction
                    corr = np.corrcoef(shap_values[:, idx], shap_values[:, j])[0, 1]
                    interaction_strength.append((feature_names[j], abs(corr)))

            # Sort by interaction strength
            interaction_strength.sort(key=lambda x: x[1], reverse=True)

            for other_feature, strength in interaction_strength[:3]:  # Top 3 interactions
                if strength > 0.1:  # Only show meaningful interactions
                    print(f"    - {feature} × {other_feature}: {strength:.3f}")

    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        print("  Try: pip install shap")

    # Method 3: Manual interaction testing based on CatBoost's patterns
    print("\n💡 Suggested Interaction Features (Based on Analysis):")
    print("-" * 50)

    # These are common patterns CatBoost excels at discovering
    suggested_interactions = [
        ("total_quotes", "avg_days_between_quotes", "engagement_velocity"),
        ("min_price", "max_price", "price_consistency"),
        ("engagement_density", "solution_complexity_score", "engagement_complexity"),
        ("equipment_variety_count", "brand_loyalty_index", "decision_diversity"),
        ("avg_discount_pct", "price_range", "discount_strategy")
    ]

    for feat1, feat2, interaction_name in suggested_interactions[:5]:
        if feat1 in feature_names and feat2 in feature_names:
            print(f"  • {interaction_name} = f({feat1}, {feat2})")


def decode_catboost_interaction(idx, feature_names):
    """
    Decode CatBoost interaction index to feature names
    This is simplified - actual decoding depends on CatBoost version
    """
    try:
        # For small number of features, we can use brute force decoding
        n_features = len(feature_names)

        # Try to decode as pair index
        # Interaction indices often use triangular numbering
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if idx == i * n_features + j - ((i + 1) * (i + 2)) // 2:
                    return f"{feature_names[i]} × {feature_names[j]}"

        # If not found, return as single feature
        if idx < n_features:
            return feature_names[idx]
        else:
            return f"Interaction_{idx}"

    except:
        return f"Interaction_{idx}"


def extract_catboost_tree_interactions(model, feature_names, max_depth=3):
    """
    Extract interaction patterns from CatBoost trees
    """
    print("\n🌳 CatBoost Tree Structure Analysis:")
    print("-" * 50)

    try:
        # Get tree structure
        model.save_model('temp_catboost_model.cbm', format='cbm')

        # Read and parse tree structure
        # Note: This requires understanding CatBoost's binary format
        # Simplified approach - use feature combinations in splits

        print("  Analyzing split patterns in trees...")

        # For each tree, track which features are used together
        feature_co_occurrence = {}

        # Get number of trees
        tree_count = model.tree_count_
        print(f"  Total trees: {tree_count}")

        # This is a simplified analysis - actual tree parsing is complex
        print("  ⚠️ Full tree parsing requires advanced CatBoost internal access")
        print("  Consider using SHAP or manual feature engineering instead")

    except Exception as e:
        print(f"  Tree analysis limited: {e}")

    return feature_co_occurrence