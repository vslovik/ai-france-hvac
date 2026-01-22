import numpy as np
import pandas as pd

WINDOW_DAYS = 90


def create_sequence_features_(df, window_days=WINDOW_DAYS):
    print("Creating sequence features (this may take a moment)...")
    print(f"  Total customers: {df['numero_compte'].nunique():,}")
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    sequence_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        # if len(customer_data) < 2:
        #     continue  # Skip single-quote customers for sequence analysis

        # Create rolling windows of quotes
        quotes = customer_data.sort_values('dt_creation_devis')

        for i in range(1, len(quotes)):
            # Look at previous quotes in the window
            current_quote = quotes.iloc[i]
            window_start = current_quote['dt_creation_devis'] - pd.Timedelta(days=window_days)
            previous_quotes = quotes.iloc[:i]
            recent_quotes = previous_quotes[previous_quotes['dt_creation_devis'] >= window_start]

            if len(recent_quotes) > 0:
                features = {
                    'numero_compte': customer_id,
                    'quote_index': i,
                    'days_since_first_quote': (
                                current_quote['dt_creation_devis'] - quotes.iloc[0]['dt_creation_devis']).days,
                    'recent_quote_count': len(recent_quotes),
                    'recent_avg_price': recent_quotes['mt_apres_remise_ht_devis'].mean(),
                    'recent_price_std': recent_quotes['mt_apres_remise_ht_devis'].std() if len(
                        recent_quotes) > 1 else 0,
                    'recent_product_variety': recent_quotes['famille_equipement_produit'].nunique(),
                    'recent_conversion_rate': recent_quotes['fg_devis_accepte'].mean(),
                    'current_price': current_quote['mt_apres_remise_ht_devis'],
                    'current_product_family': current_quote['famille_equipement_produit'],
                    'current_converted': current_quote['fg_devis_accepte']  # Target for this specific quote
                }
                sequence_features.append(features)

    df_sequence = pd.DataFrame(sequence_features)
    print(f"✓ Created {len(df_sequence):,} sequence observations")
    print(f"✓ Features include: recent patterns leading up to each quote")
    return df_sequence


def create_sequence_features(df, window_days=WINDOW_DAYS):
    print("Creating sequence features (this may take a moment)...")
    print(f"  Total customers: {df['numero_compte'].nunique():,}")
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    customer_sequence_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        quotes = customer_data.sort_values('dt_creation_devis')

        # Skip customers with no quotes (shouldn't happen, but safe)
        if len(quotes) == 0:
            continue

        # Convert to arrays
        dates = quotes['dt_creation_devis'].values
        prices = quotes['mt_apres_remise_ht_devis'].values
        converted = quotes['fg_devis_accepte'].values
        products = quotes['famille_equipement_produit'].values

        first_date = dates[0]

        # Initialize customer features - SAME STRUCTURE AS FIRST FUNCTION
        customer_features = {'numero_compte': customer_id}

        # Basic info
        customer_features['total_quotes'] = len(customer_data)
        customer_features['converted'] = customer_data['fg_devis_accepte'].max()  # Target

        # AGGREGATE THE SEQUENCE FEATURES (like first function but at customer level)
        sequence_data = []

        # Calculate features for each quote (like first function does)
        for i in range(1, len(quotes)):
            current_date = dates[i]
            window_start = current_date - np.timedelta64(window_days, 'D')

            mask = (dates[:i] >= window_start)
            recent_indices = np.where(mask)[0]

            if len(recent_indices) > 0:
                recent_prices = prices[recent_indices]
                recent_converted = converted[recent_indices]
                recent_products = products[recent_indices]

                recent_product_variety = len(pd.unique(recent_products))

                # Collect each quote's features (like first function)
                sequence_data.append({
                    'days_since_first_quote': (current_date - first_date).astype('timedelta64[D]').astype(int),
                    'recent_quote_count': len(recent_indices),
                    'recent_avg_price': np.mean(recent_prices),
                    'recent_price_std': np.std(recent_prices) if len(recent_prices) > 1 else 0,
                    'recent_product_variety': recent_product_variety,
                    'recent_conversion_rate': np.mean(recent_converted),
                    'current_price': prices[i],
                    'current_converted': converted[i]
                })

        # Now create AGGREGATE features from all sequence observations
        if len(sequence_data) > 0:
            # Convert to DataFrame for easy aggregation
            seq_df = pd.DataFrame(sequence_data)

            # Aggregate statistics (mean, std, max, etc.)
            customer_features['avg_days_since_first_quote'] = seq_df['days_since_first_quote'].mean()
            customer_features['std_days_since_first_quote'] = seq_df['days_since_first_quote'].std() if len(
                seq_df) > 1 else 0
            customer_features['max_days_since_first_quote'] = seq_df['days_since_first_quote'].max()

            customer_features['avg_recent_quote_count'] = seq_df['recent_quote_count'].mean()
            customer_features['std_recent_quote_count'] = seq_df['recent_quote_count'].std() if len(seq_df) > 1 else 0

            customer_features['avg_recent_avg_price'] = seq_df['recent_avg_price'].mean()
            customer_features['std_recent_avg_price'] = seq_df['recent_avg_price'].std() if len(seq_df) > 1 else 0

            customer_features['avg_recent_price_std'] = seq_df['recent_price_std'].mean()
            customer_features['std_recent_price_std'] = seq_df['recent_price_std'].std() if len(seq_df) > 1 else 0

            customer_features['avg_recent_product_variety'] = seq_df['recent_product_variety'].mean()
            customer_features['std_recent_product_variety'] = seq_df['recent_product_variety'].std() if len(
                seq_df) > 1 else 0

            customer_features['avg_recent_conversion_rate'] = seq_df['recent_conversion_rate'].mean()
            customer_features['std_recent_conversion_rate'] = seq_df['recent_conversion_rate'].std() if len(
                seq_df) > 1 else 0

            customer_features['avg_current_price'] = seq_df['current_price'].mean()
            customer_features['std_current_price'] = seq_df['current_price'].std() if len(seq_df) > 1 else 0

            # Ratio features
            customer_features['sequence_quote_ratio'] = len(sequence_data) / len(quotes)

            # Trend features
            if len(seq_df) > 1:
                # Price trend across sequences
                customer_features['price_trend'] = np.polyfit(range(len(seq_df)), seq_df['current_price'], 1)[0]
                # Conversion rate trend
                customer_features['conversion_rate_trend'] = \
                np.polyfit(range(len(seq_df)), seq_df['recent_conversion_rate'], 1)[0]
            else:
                customer_features['price_trend'] = 0
                customer_features['conversion_rate_trend'] = 0

        else:
            # Single quote customers - set all features to 0 or appropriate defaults
            customer_features.update({
                'avg_days_since_first_quote': 0,
                'std_days_since_first_quote': 0,
                'max_days_since_first_quote': 0,
                'avg_recent_quote_count': 0,
                'std_recent_quote_count': 0,
                'avg_recent_avg_price': 0,
                'std_recent_avg_price': 0,
                'avg_recent_price_std': 0,
                'std_recent_price_std': 0,
                'avg_recent_product_variety': 0,
                'std_recent_product_variety': 0,
                'avg_recent_conversion_rate': 0,
                'std_recent_conversion_rate': 0,
                'avg_current_price': prices[0] if len(prices) > 0 else 0,
                'std_current_price': 0,
                'sequence_quote_ratio': 0,
                'price_trend': 0,
                'conversion_rate_trend': 0
            })

        customer_sequence_features.append(customer_features)

    df_customer_sequence = pd.DataFrame(customer_sequence_features)
    print(f"✓ Created features for {len(df_customer_sequence):,} customers")
    print(f"✓ New features: {list(df_customer_sequence.columns)[:10]}...")
    return df_customer_sequence