import pandas as pd
import torch


def analyze_input_feature_importance(model, X_sample):
    """
    Analyze which INPUT features matter using gradient-based importance
    """
    model.eval()

    # Convert to tensor with gradient tracking
    X_tensor = torch.FloatTensor(X_sample.values[:100])
    X_tensor.requires_grad = True

    # Forward pass
    output = model(X_tensor)

    # Create dummy target for gradient computation
    dummy_target = torch.ones_like(output)

    # Backward pass to get gradients w.r.t inputs
    model.zero_grad()
    output.backward(dummy_target)

    # Get average absolute gradient per INPUT feature
    gradients = X_tensor.grad.abs().mean(dim=0).numpy()

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X_sample.columns.tolist(),
        'gradient_importance': gradients
    }).sort_values('gradient_importance', ascending=False)

    print("\n" + "=" * 80)
    print("GRADIENT-BASED INPUT FEATURE IMPORTANCE")
    print("=" * 80)
    print(f"\nTop 20 input features by gradient magnitude:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:40s} | Gradient: {row['gradient_importance']:.6f}")

    # Also show feature categories
    print(f"\n🔍 FEATURE CATEGORY ANALYSIS:")

    categories = {
        'Price': ['price'],
        'Quote': ['quote'],
        'Day': ['day'],
        'Average': ['avg_'],
        'Std Dev': ['std_'],
        'Trend': ['trend'],
        'Ratio': ['ratio', 'div', 'per'],
        'Log': ['log'],
        'Squared': ['squared'],
        'Tanh': ['tanh'],
        'Sqrt': ['sqrt']
    }

    for cat_name, keywords in categories.items():
        cat_features = [f for f in importance_df['feature']
                        if any(kw in f.lower() for kw in keywords)]

        if cat_features:
            cat_importance = importance_df[
                importance_df['feature'].isin(cat_features)
            ]['gradient_importance'].mean()

            print(f"  {cat_name:10s}: {len(cat_features):2d} features | "
                  f"Avg importance: {cat_importance:.6f}")

    return importance_df


def analyze_attention_weights(model, X_sample):
    """
    Analyze which HIDDEN features the attention mechanism focuses on
    """
    model.eval()

    # Get a sample batch
    if isinstance(X_sample, pd.DataFrame):
        X_tensor = torch.FloatTensor(X_sample.values[:100])  # First 100 samples
    else:
        X_tensor = torch.FloatTensor(X_sample[:100])

    with torch.no_grad():
        # Forward pass through network
        features = model.net(X_tensor)  # Shape: [100, 64] (hidden features)
        attention_weights = model.attention(features)  # Shape: [100, 64]

        # Get average attention per HIDDEN feature
        avg_attention = attention_weights.mean(dim=0).squeeze().numpy()

    print(f"Input features: {X_sample.shape[1]}")
    print(f"Hidden features: {features.shape[1]}")

    # Create importance DataFrame for HIDDEN features
    importance_df = pd.DataFrame({
        'hidden_feature_idx': list(range(len(avg_attention))),  # FIX: list of ints
        'attention_weight': avg_attention
    }).sort_values('attention_weight', ascending=False)

    print("\n" + "=" * 80)
    print("ATTENTION-BASED HIDDEN FEATURE IMPORTANCE")
    print("=" * 80)
    print(f"\nTop 20 hidden features by attention weight:")
    for i, row in importance_df.head(20).iterrows():
        # FIX: Convert to int for formatting
        feat_idx = int(row['hidden_feature_idx'])
        print(f"  Hidden feature {feat_idx:3d} | Attention: {row['attention_weight']:.4f}")

    # Check if attention is actually working
    variance = importance_df['attention_weight'].var()
    print(f"\n🔍 Attention variance: {variance:.6f}")
    if variance < 0.001:
        print("⚠️  WARNING: Attention weights are nearly identical!")
    else:
        print("✅ GOOD: Attention weights vary across features")

    return importance_df


def gradient_based_importance(model, X_sample, y_sample):
    """
    Compute feature importance using gradients (Integrated Gradients-like)
    """
    model.eval()

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_sample.values[:100])
    X_tensor.requires_grad = True

    # Forward pass
    output = model(X_tensor)

    # Create dummy target (we want gradients w.r.t inputs)
    dummy_target = torch.ones_like(output)

    # Backward pass to get gradients
    model.zero_grad()
    output.backward(dummy_target)

    # Get average absolute gradient per feature
    gradients = X_tensor.grad.abs().mean(dim=0).numpy()

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X_sample.columns.tolist(),
        'gradient_importance': gradients
    }).sort_values('gradient_importance', ascending=False)

    print("\n" + "=" * 80)
    print("GRADIENT-BASED FEATURE IMPORTANCE")
    print("=" * 80)
    print(f"\nTop 20 features by gradient magnitude:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:40s} | Gradient: {row['gradient_importance']:.6f}")

    return importance_df