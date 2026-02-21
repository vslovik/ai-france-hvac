import pandas as pd

from ml_simulation.data import Simulation
from ml_simulation.shift import ConversionShiftSimulator


class BudgetConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model):
        super().__init__(df_quotes, model)

    def apply_change(self) -> pd.DataFrame:
        """
        Switches the most recent quote per customer to a 'budget version' price,
        using tiered target prices from Simulation.PRODUCT_TIERS when available,
        with a maximum reduction cap of 30%.

        Only the most recent quote (by dt_creation_devis DESC, then id_devis DESC)
        per numero_compte is modified.

        Assumptions:
        - One row = one unique quote (id_devis unique)
        - mt_apres_remise_ht_devis is the total net price of the quote
        - famille_equipement_produit is consistent within each quote (or we take .iloc[0])
        - Simulation.PRODUCT_TIERS is accessible and contains dicts like {'p30': value}
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = [
            'numero_compte',
            'dt_creation_devis',
            'id_devis',
            'mt_apres_remise_ht_devis',
            'famille_equipement_produit'
        ]
        missing = [c for c in required if c not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Optional: enforce uniqueness (as per your earlier assertion)
        if self.df_quotes['id_devis'].duplicated().any():
            raise ValueError("id_devis is not unique — function assumes one row = one quote")

        df = self.df_quotes.copy()

        # Parse dates safely
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Sort: most recent quotes first
        df_sorted = df.sort_values(
            by=['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)

        # Identify most recent quote per customer
        df_sorted['rank'] = df_sorted.groupby('numero_compte').cumcount()
        mask_target = df_sorted['rank'] == 0

        # ── Compute target (budget) price for target quotes only ───────────────
        def get_budget_price(row):
            price = row['mt_apres_remise_ht_devis']
            product = row['famille_equipement_produit']

            if product in Simulation.PRODUCT_TIERS:
                target = Simulation.PRODUCT_TIERS[product].get('p30', price * 0.7)
            else:
                target = price * 0.7

            # Apply max 30% reduction cap
            max_reduction = 0.3
            reduction = 1 - (target / price) if price > 0 else 0
            if reduction > max_reduction:
                return price * (1 - max_reduction)
            return target

        # Apply only to target rows
        new_prices = df_sorted.apply(
            lambda row: get_budget_price(row) if row['rank'] == 0 else row['mt_apres_remise_ht_devis'],
            axis=1
        )

        df_sorted['mt_apres_remise_ht_devis'] = new_prices

        # Optional: prevent negative / zero prices (business rule choice)
        # df_sorted['mt_apres_remise_ht_devis'] = df_sorted['mt_apres_remise_ht_devis'].clip(lower=0.01)

        # Clean up helper
        df_sorted = df_sorted.drop(columns=['rank'], errors='ignore')

        # Restore original row order
        return df_sorted.set_index(df.index).sort_index()


def simulate_budget_conversion_shift(df_quotes, model):
    simulator = BudgetConversionShiftSimulator(df_quotes, model)
    return simulator.run()
