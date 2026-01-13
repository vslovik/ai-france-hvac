import pandas as pd

WINDOW_DAYS = 90


def create_sequence_features(df, window_days=WINDOW_DAYS):
    print("Creating sequence features (this may take a moment)...")
    print(f"  Total customers: {df['numero_compte'].nunique():,}")
    df = df.sort_values(['numero_compte', 'dt_creation_devis']).copy()

    sequence_features = []

    for customer_id, customer_data in df.groupby('numero_compte'):
        if len(customer_data) < 2:
            continue  # Skip single-quote customers for sequence analysis

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
