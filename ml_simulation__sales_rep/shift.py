import numpy as np
import pandas as pd

from ml_simulation.constrants import HIGH_PRICE, COLD_REGIONS
from ml_simulation.shift import ConversionShiftSimulator


class SalesRepConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model):
        super().__init__(df_quotes, model)

    def apply_change(self) -> pd.DataFrame:
        """
        Applies profile-based extra discount to the most recent quote per customer.
        Uses 'mt_apres_remise_ht_devis' directly for total_price.
        Creates missing profile columns as needed.
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        df = self.df_quotes.copy()
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Verify required price column (no detection needed)
        price_col = 'mt_apres_remise_ht_devis'
        if price_col not in df.columns:
            raise ValueError(f"Required price column '{price_col}' not found in DataFrame")

        # ── Derive missing profile columns ──────────────────────────────────────
        if 'region' not in df.columns:
            if 'nom_region' in df.columns:
                df['region'] = df['nom_region']
            else:
                df['region'] = 'Unknown'
        df['region'] = df['region'].fillna('Unknown')

        if 'has_heat_pump' not in df.columns:
            if 'famille_equipement_produit' in df.columns:
                df['has_heat_pump'] = df['famille_equipement_produit'].str.contains(
                    r'pompe à chaleur|heat ?pump', case=False, regex=True, na=False
                )
            else:
                df['has_heat_pump'] = False
        df['has_heat_pump'] = df['has_heat_pump'].fillna(False)

        if 'has_discount' not in df.columns:
            if 'mt_remise_exceptionnelle_ht' in df.columns:
                df['has_discount'] = df['mt_remise_exceptionnelle_ht'].fillna(0) > 0
            else:
                df['has_discount'] = False
        df['has_discount'] = df['has_discount'].fillna(False)

        # ── Customer-level aggregation ──────────────────────────────────────────
        customer_agg = df.groupby('numero_compte').agg(
            quote_count=pd.NamedAgg(column='id_devis', aggfunc='count'),
            total_price=pd.NamedAgg(column=price_col, aggfunc='sum'),
            region=pd.NamedAgg(column='region', aggfunc='last'),
            has_heat_pump=pd.NamedAgg(column='has_heat_pump', aggfunc='any'),
            has_discount=pd.NamedAgg(column='has_discount', aggfunc='any'),
        ).reset_index()

        # Fill defaults if any aggregation failed/missing
        customer_agg['total_price'] = customer_agg['total_price'].fillna(0.0)
        customer_agg['region'] = customer_agg['region'].fillna('Unknown')
        customer_agg['has_heat_pump'] = customer_agg['has_heat_pump'].fillna(False)
        customer_agg['has_discount'] = customer_agg['has_discount'].fillna(False)

        # ── Merge back to quotes ────────────────────────────────────────────────
        df = df.merge(customer_agg, on='numero_compte', how='left', suffixes=('', '_cust'))

        # ── Compute profile flags and score ─────────────────────────────────────
        df = df.assign(
            is_shopping_around=lambda d: d['quote_count'].fillna(0) >= 2,
            is_premium=lambda d: d['total_price'].fillna(0) > HIGH_PRICE,
            is_cold_region_heatpump=lambda d: d['region'].isin(COLD_REGIONS) & d['has_heat_pump'].fillna(False),
            current_rep=lambda d: (
                d['current_rep'].fillna('Unknown')
                if 'current_rep' in d.columns
                else pd.Series('Unknown', index=d.index)
            ),
        )

        df['score'] = (
                df['is_premium'].astype(int) +
                df['is_cold_region_heatpump'].astype(int) -
                df['has_discount'].astype(int) -
                df['is_shopping_around'].astype(int)
        )

        df['price_value_profile'] = pd.cut(
            df['score'],
            bins=[-np.inf, -1, 1, np.inf],
            labels=['price_sensitive', 'neutral', 'value_sensitive'],
            include_lowest=True
        ).astype(str)

        # ── Apply discount to most recent quote ─────────────────────────────────
        df = df.sort_values(
            ['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        )
        df['rank'] = df.groupby('numero_compte').cumcount()
        mask_target = df['rank'] == 0

        extra_pct = pd.Series(0.0, index=df.index, dtype=float)
        extra_pct[mask_target & (df['price_value_profile'] == 'price_sensitive')] = 0.025
        extra_pct[mask_target & (df['price_value_profile'] == 'value_sensitive')] = 0.006

        extra_amount = df['mt_apres_remise_ht_devis'] * extra_pct

        if 'mt_remise_exceptionnelle_ht' in df.columns:
            df['mt_remise_exceptionnelle_ht'] = df['mt_remise_exceptionnelle_ht'].fillna(0.0) + extra_amount

        df['mt_apres_remise_ht_devis'] -= extra_amount

        # Cleanup temporary columns
        temp_cols = [
            'quote_count', 'total_price', 'region', 'has_heat_pump', 'has_discount',
            'is_shopping_around', 'is_premium', 'is_cold_region_heatpump',
            'score', 'price_value_profile', 'rank'
        ]
        df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')

        return df.sort_index()


def simulate_sales_rep_conversion_shift(df_quotes, model):
    simulator = SalesRepConversionShiftSimulator(df_quotes, model)
    return simulator.run()
