import sys
import os

import numpy as np
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


def compute_product_statistics(
        df_train: pd.DataFrame,
        group_by_col: str = 'famille_equipement_produit',
        price_col: str = 'mt_apres_remise_ht_devis',
        margin_col: str = 'mt_marge',
        accepted_col: str = 'fg_devis_accepte',
        output_path: str = None  # optional: save to pickle/json/csv
) -> dict:
    """
    Computes product-level statistics from the training dataset.

    Returns a dictionary keyed by product (famille_equipement_produit),
    with statistics that can be used as global constants in transformation logic.

    Example output structure:
    {
        'Pompe à chaleur air/air': {
            'count': 342,
            'avg_price': 4850.75,
            'median_price': 4600.0,
            'min_price': 3200.0,
            'max_price': 7800.0,
            'std_price': 920.4,
            'avg_margin_pct': 0.312,
            'acceptance_rate': 0.68,
            'p30': 4550.0,          # e.g. 30th percentile or business-defined target
            ...
        },
        'Chaudière gaz': { ... },
        ...
    }

    Optionally saves the result to disk for reuse.
    """
    if group_by_col not in df_train.columns:
        raise ValueError(f"Grouping column '{group_by_col}' not found in df_train")
    if price_col not in df_train.columns:
        raise ValueError(f"Price column '{price_col}' not found in df_train")

    # Basic filtering (optional - customize)
    df = df_train[df_train[price_col] > 0].copy()

    # Core aggregations
    stats = df.groupby(group_by_col).agg(
        count=(price_col, 'count'),
        avg_price=(price_col, 'mean'),
        median_price=(price_col, 'median'),
        min_price=(price_col, 'min'),
        max_price=(price_col, 'max'),
        std_price=(price_col, 'std'),
        p10=(price_col, lambda x: x.quantile(0.10)),
        p30=(price_col, lambda x: x.quantile(0.30)),  # often used as budget/target
        p50=(price_col, 'median'),
        p70=(price_col, lambda x: x.quantile(0.70)),
        p90=(price_col, lambda x: x.quantile(0.90)),
    )

    # Margin percentage (if margin column exists)
    if margin_col in df.columns:
        df['margin_pct'] = df[margin_col] / df[price_col].replace(0, np.nan)
        margin_agg = df.groupby(group_by_col)['margin_pct'].mean().rename('avg_margin_pct')
        stats = stats.join(margin_agg)

    # Acceptance rate (if flag exists)
    if accepted_col in df.columns:
        acceptance = df.groupby(group_by_col)[accepted_col].mean().rename('acceptance_rate')
        stats = stats.join(acceptance)

    # Convert to clean dictionary
    product_stats = stats.reset_index().set_index(group_by_col).to_dict(orient='index')

    # Round numbers for readability & storage
    for prod, values in product_stats.items():
        for k, v in values.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if k.endswith('_pct') or k.startswith('acceptance'):
                    values[k] = round(v, 4)
                else:
                    values[k] = round(v, 2)

    # Optional: save for reuse across sessions/scripts
    if output_path:
        if output_path.endswith('.json'):
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(product_stats, f, ensure_ascii=False, indent=2)
        elif output_path.endswith('.pkl'):
            pd.to_pickle(product_stats, output_path)
        else:
            stats.to_csv(output_path)

    return product_stats
