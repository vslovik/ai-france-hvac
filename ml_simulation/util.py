import sys
import os
import pandas as pd


# Suppress create_features output
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._null = open(os.devnull, 'w')
        sys.stdout = self._null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        # Ensure the file is actually closed
        if not self._null.closed:
            self._null.close()


def get_product_price_tiers(df_simulation):
    # 5. Get price percentiles per product from SIM pool
    product_prices = {}
    price_stats = []

    for product in df_simulation['famille_equipement_produit'].unique():
        prices = df_simulation[df_simulation['famille_equipement_produit'] == product][
            'mt_apres_remise_ht_devis'].dropna()
        if len(prices) >= 5:
            p30 = prices.quantile(0.3)
            p70 = prices.quantile(0.7)
            p90 = prices.quantile(0.9)
            product_prices[product] = {'p30': p30, 'p70': p70, 'p90': p90}
            price_stats.append({
                'product': product,
                'count': len(prices),
                'p30': p30,
                'p70': p70,
                'p90': p90
            })

    print(f"Products with price data: {len(product_prices)}")

    # Show price stats
    df_stats = pd.DataFrame(price_stats).sort_values('count', ascending=False)
    print("\n📊 PRODUCT PRICE TIERS (from simulation pool):")
    print(df_stats.to_string(index=False))

    return product_prices


