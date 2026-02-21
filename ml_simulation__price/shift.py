import pandas as pd

from ml_simulation.shift import ConversionShiftSimulator


class PriceConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, price_change_percentage):
        super().__init__(df_quotes, model)
        self.price_change_percentage = price_change_percentage

    def apply_change(self) -> pd.DataFrame:
        """
        Increases or decreases the net price (mt_apres_remise_ht_devis)
        of the MOST RECENT quote per customer by the given percentage.

        Parameters:
        -----------
        df_quotes : pd.DataFrame
            DataFrame with one row per quote (id_devis unique)
        price_change_percentage : float
            Percentage change (e.g. 0.10 = +10%, -0.15 = -15%)

        Behavior:
        - Only the most recent quote per numero_compte is modified
        - Recency: highest dt_creation_devis → on tie: highest id_devis
        - Other quotes remain unchanged
        - Works with empty DataFrame (returns copy)
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = ['numero_compte', 'dt_creation_devis', 'id_devis', 'mt_apres_remise_ht_devis']
        missing = [c for c in required if c not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Defensive: check uniqueness if this is important for your use case
        if self.df_quotes['id_devis'].duplicated().any():
            raise ValueError("id_devis is not unique — function assumes one row = one quote")

        df = self.df_quotes.copy()

        # Parse dates safely
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Sort: most recent quotes come first
        df_sorted = df.sort_values(
            by=['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)

        # Identify the row to modify (rank 0 = most recent per customer)
        df_sorted['rank'] = df_sorted.groupby('numero_compte').cumcount()
        mask_target = df_sorted['rank'] == 0

        # Calculate new price only for target rows
        current_price = df_sorted['mt_apres_remise_ht_devis']
        new_price = current_price * (1 + self.price_change_percentage)

        # Apply only to selected rows
        df_sorted['mt_apres_remise_ht_devis'] = current_price.where(
            ~mask_target,  # keep original when not target
            new_price  # apply new value when target
        )

        # Optional: prevent negative prices (uncomment if desired)
        # df_sorted['mt_apres_remise_ht_devis'] = df_sorted['mt_apres_remise_ht_devis'].clip(lower=0)

        # Clean up helper column
        df_sorted = df_sorted.drop(columns=['rank'], errors='ignore')

        # Restore original row order
        return df_sorted.set_index(df.index).sort_index()


def simulate_price_conversion_shift(df_quotes, model, price_change_percentage):
    simulator = PriceConversionShiftSimulator(df_quotes, model, price_change_percentage)
    return simulator.run()