def create_price_binning_features(df, price_cols=None, n_bins=5):
    """
    Create binned versions of price features to capture non-linear patterns.
    Automatically detects sweet spots in the price-conversion relationship.

    Args:
        df: DataFrame with features
        price_cols: List of price columns to bin (if None, uses common price columns)
        n_bins: Number of bins to create (default 5)
    """
    import pandas as pd
    import numpy as np

    df_new = df.copy()

    print("\n" + "=" * 80)
    print("📊 CREATING PRICE BINNING FEATURES (Sweet Spot Detection)")
    print("=" * 80)

    initial_cols = set(df_new.columns)

    # Default price columns if none provided
    if price_cols is None:
        price_cols = ['max_price', 'avg_current_price', 'min_price', 'avg_price',
                      'value_score', 'price_x_sophistication']
        # Only use columns that exist
        price_cols = [col for col in price_cols if col in df_new.columns]

    print(f"\nBinning {len(price_cols)} price features...")

    for col in price_cols:
        if col in df_new.columns:
            # Remove infinite and NaN values
            valid_data = df_new[col].replace([np.inf, -np.inf], np.nan).dropna()

            if len(valid_data) > n_bins * 10:  # Enough data for binning

                # 1. Quantile bins (equal frequency)
                try:
                    df_new[f'{col}_bin_quantile'] = pd.qcut(
                        df_new[col],
                        q=n_bins,
                        labels=[f'Q{i + 1}' for i in range(n_bins)],
                        duplicates='drop'
                    ).astype(str)
                except:
                    # Fallback to equal width bins if quantile fails
                    df_new[f'{col}_bin_width'] = pd.cut(
                        df_new[col],
                        bins=n_bins,
                        labels=[f'Bin{i + 1}' for i in range(n_bins)]
                    ).astype(str)

                # 2. Business-meaningful bins (capture sweet spots)
                percentiles = df_new[col].quantile([0.2, 0.4, 0.6, 0.8]).values

                # Create custom labels that reflect business value
                labels = ['Very Low', 'Low', 'Medium-Low', 'Medium-High', 'High', 'Very High']
                if n_bins <= 6:
                    labels = labels[:n_bins]

                df_new[f'{col}_tier'] = pd.cut(
                    df_new[col],
                    bins=[-np.inf] + list(percentiles) + [np.inf],
                    labels=labels[:len(percentiles) + 1]
                ).astype(str)

                # 3. SWEET SPOT DETECTION - This is key!
                # If we have conversion data, find actual sweet spots
                if 'converted' in df_new.columns:
                    # Calculate conversion rate by price bucket
                    temp_df = df_new[[col, 'converted']].copy()
                    temp_df['price_bucket'] = pd.cut(temp_df[col], bins=20)

                    # Find buckets with highest conversion
                    conversion_by_bucket = temp_df.groupby('price_bucket')['converted'].mean()
                    top_buckets = conversion_by_bucket.nlargest(3).index

                    # Create sweet spot indicator
                    df_new[f'{col}_sweet_spot'] = 0
                    for bucket in top_buckets:
                        mask = (temp_df['price_bucket'] == bucket).values
                        df_new.loc[mask, f'{col}_sweet_spot'] = 1

                # 4. Create one-hot encoded versions for tree models
                dummies = pd.get_dummies(df_new[f'{col}_tier'], prefix=f'{col}_tier')
                df_new = pd.concat([df_new, dummies], axis=1)

    # Special focus on your top price features
    top_price_features = ['max_price', 'avg_current_price']
    for col in top_price_features:
        if col in df_new.columns:
            # Create more granular bins for top features
            try:
                df_new[f'{col}_bin_10'] = pd.qcut(
                    df_new[col],
                    q=10,
                    labels=[f'Decile{i + 1}' for i in range(10)],
                    duplicates='drop'
                ).astype(str)
            except:
                pass

    # Create interaction between binned price and consistency
    if all(col in df_new.columns for col in ['max_price_tier', 'quote_consistency_score']):
        df_new['price_tier_x_consistency'] = (
                df_new['max_price_tier'].astype(str) + "_x_" +
                pd.cut(df_new['quote_consistency_score'], bins=3, labels=['Low', 'Med', 'High']).astype(str)
        )

    # Print summary
    new_features = set(df_new.columns) - initial_cols
    print(f"\n✅ Created {len(new_features)} price binning features")
    print("\n📊 BINNING TYPES CREATED:")
    print("  • Quantile bins (equal frequency)")
    print("  • Business tier bins (Very Low to Very High)")
    print("  • Sweet spot indicators (detected from conversion rates)")
    print("  • One-hot encoded dummies")
    print("  • Interactions with consistency")

    if 'converted' in df_new.columns:
        print("\n🎯 SWEET SPOT DETECTION ACTIVE - Finding optimal price ranges")

    return df_new