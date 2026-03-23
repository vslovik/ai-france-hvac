# %% [markdown]
# ## Standardized Pricing Simulation

# %%
import pandas as pd
import numpy as np
from ml_simulation.shift import ConversionShiftSimulator


class StandardizedPricingConversionShiftSimulator(ConversionShiftSimulator):
    """
    Simulates conversion shift by replacing current prices with standardized prices.

    Only overrides apply_change() - parent class handles all feature creation and prediction.
    """

    def __init__(self, df_quotes, model, feature_names, price_file, variant='moy_pv', show_plot=True,
                 log_to_wandb=False):
        """
        Args:
            df_quotes: DataFrame with customer quotes
            model: Trained XGBoost model
            feature_names: List of feature names the model expects (from model_data)
            price_file: DataFrame with standardized prices (Liste_des_devis_type.csv)
            variant: 'min_pv', 'max_pv', or 'moy_pv' - which price to use
            show_plot: Whether to display plots
            log_to_wandb: Whether to log to Weights & Biases
        """
        super().__init__(df_quotes, model, show_plot, log_to_wandb)
        self.feature_names = feature_names
        self.price_file = price_file
        self.variant = variant

        # Build price mapping
        self.price_map = self._build_price_map()

    def _build_price_map(self):
        """
        Build mapping ONLY for exact product matches.
        No fallbacks - if exact match not found, skip.
        """
        price_col = self.variant

        price_map = {}

        for _, row in self.price_file.iterrows():
            # Skip if any required field is missing
            if pd.isna(row['famille_equipement_produit']) or pd.isna(row['type_equipement_produit']):
                continue

            key = (
                row['famille_equipement_produit'],
                row['type_equipement_produit'],
                row['marque_produit'] if pd.notna(row['marque_produit']) else ''
            )
            price_map[key] = row[price_col]

        print(f"✅ Built price mapping with {len(price_map)} exact product matches")
        return price_map

    def apply_change(self) -> pd.DataFrame:
        """
        Apply standardized pricing ONLY to quotes that have an exact match.
        Quotes without exact match keep their original price.
        """
        df = self.df_quotes.copy()

        modified_count = 0
        skipped_count = 0

        for idx, quote in df.iterrows():
            product_family = quote.get('famille_equipement_produit', '')
            product_type = quote.get('type_equipement_produit', '')
            product_brand = quote.get('marque_produit', '') if pd.notna(quote.get('marque_produit', '')) else ''

            key = (product_family, product_type, product_brand)

            if key in self.price_map:
                # Replace price with standardized price
                df.loc[idx, 'mt_apres_remise_ht_devis'] = self.price_map[key]
                modified_count += 1
            else:
                skipped_count += 1
                # Keep original price

        print(f"✅ Applied standardized pricing ({self.variant}):")
        print(f"   - Modified quotes: {modified_count}")
        print(f"   - Skipped quotes (no exact match): {skipped_count}")

        return df


def simulate_standardized_pricing_conversion_shift(
        df_quotes,
        model,
        feature_names,
        price_file,
        variant='moy_pv',
        show_plot=True,
        log_to_wandb=False
):
    """
    Wrapper function to simulate conversion shift with standardized pricing.

    Args:
        df_quotes: DataFrame with customer quotes
        model: Trained XGBoost model
        feature_names: List of feature names the model expects (from model_data)
        price_file: DataFrame with standardized prices
        variant: 'min_pv', 'max_pv', or 'moy_pv'
        show_plot: Whether to display plots
        log_to_wandb: Whether to log to Weights & Biases

    Returns:
        DataFrame with baseline and standardized predictions
    """
    simulator = StandardizedPricingConversionShiftSimulator(
        df_quotes,
        model,
        feature_names,
        price_file,
        variant,
        show_plot,
        log_to_wandb
    )
    return simulator.run()
