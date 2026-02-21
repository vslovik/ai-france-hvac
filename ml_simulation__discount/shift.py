import pandas as pd

from ml_simulation.shift import ConversionShiftSimulator


class DiscountConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, discount_percentage):
        super().__init__(df_quotes, model)
        self.discount_percentage = discount_percentage

    def apply_change(self) -> pd.DataFrame:
        """
        Applies a discount ONLY to the most recent quote per customer (numero_compte).

        Recency rule:
          - Highest dt_creation_devis first
          - If dates are equal → highest id_devis wins

        Preconditions (assumed / enforced):
          - Each row = one unique quote (id_devis is unique across the whole dataframe)
          - mt_apres_remise_ht_devis = total net price of that quote (no line items)

        The discount is applied directly to that single row:
          - added to mt_remise_exceptionnelle_ht
          - subtracted from mt_apres_remise_ht_devis
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = ['numero_compte', 'dt_creation_devis', 'id_devis', 'mt_apres_remise_ht_devis']
        missing = [c for c in required if c not in self.df_quotes]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Defensive check based on your assertion
        if self.df_quotes['id_devis'].duplicated().any():
            raise ValueError("id_devis is not unique — the function assumes one row = one quote")

        df = self.df_quotes.copy()

        # Prepare sorting keys
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Sort: most recent quotes first (date desc + id desc on ties)
        df_sorted = df.sort_values(
            by=['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)

        # Identify the most recent quote per customer
        df_sorted['rank'] = df_sorted.groupby('numero_compte').cumcount()
        mask_target = df_sorted['rank'] == 0

        # Calculate discount amount — only for target rows
        discount_amount = df_sorted['mt_apres_remise_ht_devis'] * (self.discount_percentage / 100.0)
        discount_amount = discount_amount.where(mask_target, 0.0)

        # Apply changes — only to the selected quotes
        if 'mt_remise_exceptionnelle_ht' in df_sorted.columns:
            current_remise = df_sorted['mt_remise_exceptionnelle_ht'].fillna(0.0)
            df_sorted['mt_remise_exceptionnelle_ht'] = current_remise + discount_amount

        df_sorted['mt_apres_remise_ht_devis'] = (
                df_sorted['mt_apres_remise_ht_devis'] - discount_amount
        )

        # Optional: prevent negative prices (business rule choice)
        # df_sorted['mt_apres_remise_ht_devis'] = df_sorted['mt_apres_remise_ht_devis'].clip(lower=0)

        # Clean up temporary columns
        df_sorted = df_sorted.drop(columns=['rank'], errors='ignore')

        # Return in original row order
        return df_sorted.set_index(df.index).sort_index()


def simulate_discount_conversion_shift(df_quotes, model, discount_percentage):
    simulator = DiscountConversionShiftSimulator(df_quotes, model, discount_percentage)
    return simulator.run()