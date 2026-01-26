import pandas as pd
import numpy as np


def create_market_features(df, brand_features_df=None):
    """
    PILLAR 1.3: Market Position Signals

    Creates features that capture brand market position and customer alignment:
    1. Brand popularity and market share alignment
    2. Niche vs mainstream brand preference
    3. Brand-customer fit signals
    4. Market concentration metrics
    """
    print("=" * 80)
    print("CREATING MARKET POSITION FEATURES")
    print("=" * 80)

    # Check if brand data is available
    if 'marque_produit' not in df.columns:
        print("⚠️ WARNING: 'marque_produit' column not found in dataset")
        print("  Returning empty market position features DataFrame")
        return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing market position data for {df['numero_compte'].nunique():,} customers")
    print(f"Total quotes with brand info: {df['marque_produit'].notna().sum():,}")

    # Sort by customer and date for consistency
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    # ========== PRE-CALCULATE GLOBAL MARKET METRICS ==========

    # Calculate overall brand statistics
    brand_counts_total = df['marque_produit'].value_counts()
    total_quotes_with_brands = df['marque_produit'].notna().sum()

    # Market share dictionary
    brand_market_share = (brand_counts_total / total_quotes_with_brands).to_dict()

    # Brand conversion rates (if conversion data available)
    brand_conversion_rates = {}
    if 'fg_devis_accepte' in df.columns:
        for brand in brand_counts_total.index:
            brand_mask = df['marque_produit'] == brand
            if brand_mask.sum() >= 5:  # Minimum 5 samples for reliability
                conversion_rate = df.loc[brand_mask, 'fg_devis_accepte'].mean()
                brand_conversion_rates[brand] = conversion_rate

    # Define market segments
    top_10_brands = brand_counts_total.head(10).index.tolist()
    top_5_brands = brand_counts_total.head(5).index.tolist()

    # Premium brand list (based on typical market knowledge)
    premium_brands = ['MITSUBISHI ELECTRIC', 'VIESSMANN', 'BOSCH', 'DE DIETRICH',
                      'BUDERUS', 'JUNKERS', 'WOLF', 'WESTEN']

    # Budget brand list
    budget_brands = ['ATLANTIC', 'FRISQUET', 'CHAPPEE', 'SAUNIER DUVAL',
                     'PROTHERM', 'THERMOR', 'AQUATIS']

    market_position_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        brand_data = customer_data['marque_produit'].dropna()

        if len(brand_data) == 0:
            # No brand data for this customer
            features.update({
                'market_data_available': 0,
                'brand_popularity_alignment': 0,
                'prefers_niche_brands': 0,
                'brand_conversion_alignment': 0.4,  # Neutral default
                'market_share_concentration': 0,
                'brand_customer_fit_score': 0.5,
                'prestige_brand_ratio': 0,
                'brand_diversification_index': 0,
                'market_leader_alignment': 0,
                'top_5_brand_alignment': 0
            })
        else:
            features['market_data_available'] = 1

            # ========== FEATURE 1: BRAND POPULARITY ALIGNMENT ==========
            # How well does customer's brand preference align with market popularity?
            popularity_scores = []
            for brand in brand_data:
                market_share = brand_market_share.get(brand, 0)
                popularity_scores.append(market_share)

            features['brand_popularity_alignment'] = np.mean(popularity_scores) if popularity_scores else 0
            features['brand_popularity_variance'] = np.var(popularity_scores) if len(popularity_scores) > 1 else 0

            # ========== FEATURE 2: NICHE VS MAINSTREAM PREFERENCE ==========
            # Count occurrences in different market segments
            in_top_10 = sum(1 for brand in brand_data if brand in top_10_brands)
            in_top_5 = sum(1 for brand in brand_data if brand in top_5_brands)
            in_premium = sum(1 for brand in brand_data if brand in premium_brands)
            in_budget = sum(1 for brand in brand_data if brand in budget_brands)

            total_brand_mentions = len(brand_data)

            features['top_10_brand_ratio'] = in_top_10 / total_brand_mentions if total_brand_mentions > 0 else 0
            features['top_5_brand_ratio'] = in_top_5 / total_brand_mentions if total_brand_mentions > 0 else 0
            features['premium_brand_ratio'] = in_premium / total_brand_mentions if total_brand_mentions > 0 else 0
            features['budget_brand_ratio'] = in_budget / total_brand_mentions if total_brand_mentions > 0 else 0

            # Niche brand indicator (not in top 10 or special lists)
            niche_count = total_brand_mentions - in_top_10
            features['prefers_niche_brands'] = niche_count / total_brand_mentions if total_brand_mentions > 0 else 0

            # ========== FEATURE 3: BRAND CONVERSION ALIGNMENT ==========
            # Does customer prefer brands with high conversion rates?
            if brand_conversion_rates:
                conversion_scores = []
                for brand in brand_data:
                    # Use actual conversion rate if available, else neutral (0.4)
                    conv_rate = brand_conversion_rates.get(brand, 0.4)
                    conversion_scores.append(conv_rate)

                features['brand_conversion_alignment'] = np.mean(conversion_scores) if conversion_scores else 0.4
                features['brand_conversion_consistency'] = np.std(conversion_scores) if len(
                    conversion_scores) > 1 else 0
            else:
                features['brand_conversion_alignment'] = 0.4  # Neutral default
                features['brand_conversion_consistency'] = 0

            # ========== FEATURE 4: MARKET SHARE CONCENTRATION ==========
            # Herfindahl-Hirschman Index (HHI) for brand concentration
            brand_counts_customer = brand_data.value_counts()
            if len(brand_counts_customer) > 0:
                proportions = brand_counts_customer / len(brand_data)
                hhi = np.sum(proportions ** 2)
                features['market_share_concentration'] = hhi
                features['effective_brands'] = 1 / hhi if hhi > 0 else 1
            else:
                features['market_share_concentration'] = 0
                features['effective_brands'] = 1

            # ========== FEATURE 5: BRAND-CUSTOMER FIT SCORE ==========
            # Composite score of market position alignment
            fit_components = []

            # 1. Popularity alignment (weighted)
            if features['brand_popularity_alignment'] > 0:
                # Normalize: typical market share ~0.02-0.18, so multiply by 5 to get 0.1-0.9 range
                normalized_popularity = min(features['brand_popularity_alignment'] * 5, 1)
                fit_components.append(normalized_popularity)

            # 2. Conversion alignment (already 0-1 scale)
            fit_components.append(features['brand_conversion_alignment'])

            # 3. Premium brand preference (premium often indicates serious buyer)
            fit_components.append(features['premium_brand_ratio'])

            # 4. Brand loyalty (if available from brand_features_df)
            if brand_features_df is not None:
                customer_mask = brand_features_df['numero_compte'] == customer_id
                if customer_mask.any():
                    customer_brand_row = brand_features_df[customer_mask]
                    if 'brand_loyalty_index' in customer_brand_row.columns:
                        loyalty = customer_brand_row['brand_loyalty_index'].iloc[0]
                        fit_components.append(loyalty)

            features['brand_customer_fit_score'] = np.mean(fit_components) if fit_components else 0.5

            # ========== FEATURE 6: MARKET LEADER ALIGNMENT ==========
            # Is customer aligned with the market leader?
            if len(brand_counts_total) > 0:
                market_leader = brand_counts_total.index[0]
                leader_count = sum(1 for brand in brand_data if brand == market_leader)
                features[
                    'market_leader_alignment'] = leader_count / total_brand_mentions if total_brand_mentions > 0 else 0
                features['uses_market_leader'] = 1 if leader_count > 0 else 0
            else:
                features['market_leader_alignment'] = 0
                features['uses_market_leader'] = 0

            # ========== FEATURE 7: BRAND DIVERSIFICATION INDEX ==========
            # Inverse of concentration - measures brand diversification
            features['brand_diversification_index'] = 1 - features['market_share_concentration']

            # ========== FEATURE 8: BRAND STRATEGY CONSISTENCY ==========
            # Is customer consistent in their brand strategy?
            strategy_components = []

            # Premium consistency
            if features['premium_brand_ratio'] > 0.7 or features['premium_brand_ratio'] < 0.3:
                strategy_components.append(1)  # Clear preference
            else:
                strategy_components.append(0)  # Mixed strategy

            # Mainstream consistency
            if features['top_10_brand_ratio'] > 0.7 or features['prefers_niche_brands'] > 0.7:
                strategy_components.append(1)
            else:
                strategy_components.append(0)

            features['brand_strategy_consistency'] = np.mean(strategy_components) if strategy_components else 0

            # ========== FEATURE 9: BRAND VALUE PROPOSITION ==========
            # Composite of premium/mainstream/conversion alignment
            value_components = [
                features['premium_brand_ratio'] * 0.4,  # Premium weight
                features['top_5_brand_ratio'] * 0.3,  # Mainstream weight
                features['brand_conversion_alignment'] * 0.3  # Performance weight
            ]
            features['brand_value_proposition'] = np.sum(value_components)

        market_position_features.append(features)

    # Convert to DataFrame
    market_features_df = pd.DataFrame(market_position_features)

    # Report statistics
    print(f"\n✅ Created {len(market_features_df.columns) - 1} market position features")
    print(f"   Samples: {len(market_features_df):,}")

    # Show feature distribution summary
    if len(market_features_df) > 0:
        numeric_features = [col for col in market_features_df.columns
                            if col != 'numero_compte' and
                            market_features_df[col].dtype in ['int64', 'float64']]

        print("\n📊 MARKET POSITION FEATURES SUMMARY:")
        print("-" * 60)
        for feat in numeric_features[:12]:  # Show first 12 features
            mean_val = market_features_df[feat].mean()
            std_val = market_features_df[feat].std()
            non_zero = (market_features_df[feat] != 0).mean() * 100
            print(f"{feat:35} : mean={mean_val:.3f}, std={std_val:.3f}, non-zero={non_zero:.1f}%")

    return market_features_df