import pandas as pd
import numpy as np
from collections import Counter


def create_brand_features(df):
    """
    PILLAR 1: Product Brand Intelligence Features
    Step 1.1: Brand Loyalty & Switching Patterns

    Creates features from 'marque_produit' column to capture:
    1. Brand loyalty metrics
    2. Brand switching behavior
    3. Brand preference patterns
    """
    print("=" * 80)
    print("CREATING BRAND INTELLIGENCE FEATURES")
    print("=" * 80)

    # Check if brand data is available
    if 'marque_produit' not in df.columns:
        print("⚠️ WARNING: 'marque_produit' column not found in dataset")
        print("  Returning empty brand features DataFrame")
        return pd.DataFrame(columns=['numero_compte'])

    print(f"Processing brand data for {df['numero_compte'].nunique():,} customers")
    print(f"Total quotes with brand info: {df['marque_produit'].notna().sum():,}")
    print(f"Unique brands in dataset: {df['marque_produit'].nunique():,}")

    # Sort by customer and date for sequence analysis
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    brand_features = []

    # Pre-calculate brand market share for later use
    brand_counts = df['marque_produit'].value_counts()
    total_quotes = len(df)
    brand_market_share = (brand_counts / total_quotes).to_dict()

    # Define brand tiers based on market share (for premium/budget classification)
    top_brands = brand_counts.head(10).index.tolist()  # Top 10 by frequency
    premium_brands = ['MITSUBISHI ELECTRIC', 'VIESSMANN', 'BOSCH', 'DE DIETRICH', 'BUDERUS']
    budget_brands = ['ATLANTIC', 'FRISQUET', 'CHAPPEE', 'SAUNIER DUVAL', 'PROTHERM']

    for customer_id, customer_data in df.groupby('numero_compte'):
        features = {'numero_compte': customer_id}

        brand_data = customer_data['marque_produit'].dropna()

        if len(brand_data) == 0:
            # No brand data for this customer
            features.update({
                'brand_data_available': 0,
                'brand_loyalty_index': 0,
                'brand_switches': 0,
                'prefers_premium_brand': 0,
                'prefers_budget_brand': 0,
                'top_brand_share': 0,
                'brand_exploration_score': 0,
                'market_share_weighted_brand_score': 0,
                'brand_consistency': 0
            })
        else:
            features['brand_data_available'] = 1

            # ========== FEATURE 1: BRAND LOYALTY INDEX ==========
            # Measures how loyal customer is to one brand (0-1 scale)
            brand_counts_customer = brand_data.value_counts()
            top_brand_count = brand_counts_customer.iloc[0] if len(brand_counts_customer) > 0 else 0
            total_brand_mentions = len(brand_data)
            features['brand_loyalty_index'] = top_brand_count / total_brand_mentions

            # ========== FEATURE 2: BRAND SWITCHES ==========
            # Number of different brands considered
            unique_brands = brand_data.nunique()
            features['brand_switches'] = unique_brands - 1  # Zero for single brand

            # ========== FEATURE 3: BRAND TIER PREFERENCE ==========
            top_brand = brand_counts_customer.index[0] if len(brand_counts_customer) > 0 else None

            # Premium vs Budget classification
            features['prefers_premium_brand'] = 1 if top_brand in premium_brands else 0
            features['prefers_budget_brand'] = 1 if top_brand in budget_brands else 0

            # Top 10 brand indicator
            features['prefers_top_10_brand'] = 1 if top_brand in top_brands else 0

            # ========== FEATURE 4: TOP BRAND SHARE ==========
            # What percentage of their quotes are for their preferred brand
            features['top_brand_share'] = features['brand_loyalty_index']

            # ========== FEATURE 5: BRAND EXPLORATION SCORE ==========
            # Measures how much they explore different brands vs sticking to one
            # 0 = always same brand, 1 = maximum variety
            max_possible_variety = min(len(brand_data), len(brand_counts))
            if max_possible_variety > 1:
                features['brand_exploration_score'] = (unique_brands - 1) / (max_possible_variety - 1)
            else:
                features['brand_exploration_score'] = 0

            # ========== FEATURE 6: MARKET SHARE WEIGHTED BRAND SCORE ==========
            # Weight brands by their overall market popularity
            brand_weights = []
            for brand in brand_data:
                brand_weights.append(brand_market_share.get(brand, 0))
            features['market_share_weighted_brand_score'] = np.mean(brand_weights) if brand_weights else 0

            # ========== FEATURE 7: BRAND CONSISTENCY ==========
            # Binary: 1 if all quotes same brand, 0 otherwise
            features['brand_consistency'] = 1 if unique_brands == 1 else 0

            # ========== FEATURE 8: BRAND SEQUENCE ANALYSIS ==========
            # For customers with multiple quotes: analyze brand switching patterns
            if len(brand_data) > 1:
                # Convert to list for sequence analysis
                brand_sequence = brand_data.tolist()

                # Brand persistence: how often they stick with same brand
                brand_changes = sum(1 for i in range(1, len(brand_sequence))
                                    if brand_sequence[i] != brand_sequence[i - 1])
                features['brand_persistence_ratio'] = 1 - (brand_changes / (len(brand_sequence) - 1))

                # Brand momentum: are they converging on a brand?
                if unique_brands > 1:
                    first_half = brand_sequence[:len(brand_sequence) // 2]
                    second_half = brand_sequence[len(brand_sequence) // 2:]
                    first_top = Counter(first_half).most_common(1)[0][0] if first_half else None
                    second_top = Counter(second_half).most_common(1)[0][0] if second_half else None
                    features['brand_convergence'] = 1 if first_top == second_top else 0
                else:
                    features['brand_convergence'] = 1  # Already converged
            else:
                features['brand_persistence_ratio'] = 1  # Single quote = perfect persistence
                features['brand_convergence'] = 1

        brand_features.append(features)

    # Convert to DataFrame
    brand_features_df = pd.DataFrame(brand_features)

    # Report statistics
    print(f"\n✅ Created {len(brand_features_df.columns) - 1} brand intelligence features")
    print(f"   Samples: {len(brand_features_df):,}")

    # Show feature distribution summary
    if len(brand_features_df) > 0:
        numeric_features = [col for col in brand_features_df.columns
                            if col != 'numero_compte' and brand_features_df[col].dtype in ['int64', 'float64']]

        print("\n📊 BRAND FEATURES SUMMARY:")
        print("-" * 50)
        for feat in numeric_features[:10]:  # Show first 10 features
            mean_val = brand_features_df[feat].mean()
            std_val = brand_features_df[feat].std()
            print(f"{feat:30} : mean={mean_val:.3f}, std={std_val:.3f}")

        if len(numeric_features) > 10:
            print(f"... and {len(numeric_features) - 10} more features")

    return brand_features_df