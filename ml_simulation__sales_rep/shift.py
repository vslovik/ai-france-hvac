import pandas as pd

from ml_simulation.constrants import HIGH_PRICE, COLD_REGIONS
from ml_simulation.shift import ConversionShiftSimulator


class SalesRepConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model):
        super().__init__(df_quotes, model)

    def apply_change(self) -> pd.DataFrame:
        """
        Applies profile-based extra discount to the most recent quote per customer.
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        df = self.df_quotes.copy()
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Verify required price column
        price_col = 'mt_apres_remise_ht_devis'
        if price_col not in df.columns:
            raise ValueError(f"Required price column '{price_col}' not found")

        # ── Customer-level aggregation ──────────────────────────────────────────
        customer_agg = df.groupby('numero_compte').agg(
            quote_count=pd.NamedAgg(column='id_devis', aggfunc='count'),
            total_price=pd.NamedAgg(column=price_col, aggfunc='sum'),
            has_discount=pd.NamedAgg(
                column='mt_remise_exceptionnelle_ht',
                aggfunc=lambda x: (x.fillna(0) > 0).any()
            ) if 'mt_remise_exceptionnelle_ht' in df.columns else pd.NamedAgg(column='id_devis',
                                                                              aggfunc=lambda x: False),
            region=pd.NamedAgg(
                column='nom_region',
                aggfunc='last'
            ) if 'nom_region' in df.columns else pd.NamedAgg(column='id_devis', aggfunc=lambda x: 'Unknown'),
            has_heat_pump=pd.NamedAgg(
                column='famille_equipement_produit',
                aggfunc=lambda x: (x == 'Pompe à chaleur').any()
            ) if 'famille_equipement_produit' in df.columns else pd.NamedAgg(column='id_devis',
                                                                             aggfunc=lambda x: False),
        ).reset_index()

        # Fill defaults
        customer_agg['total_price'] = customer_agg['total_price'].fillna(0)
        customer_agg['region'] = customer_agg.get('region', pd.Series('Unknown', index=customer_agg.index))
        customer_agg['has_discount'] = customer_agg.get('has_discount', pd.Series(False, index=customer_agg.index))
        customer_agg['has_heat_pump'] = customer_agg.get('has_heat_pump', pd.Series(False, index=customer_agg.index))

        # ── Merge back to quotes ────────────────────────────────────────────────
        df = df.merge(customer_agg, on='numero_compte', how='left', suffixes=('', '_cust'))

        # Find most recent quote per customer
        df = df.sort_values(
            ['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        )
        df['rank'] = df.groupby('numero_compte').cumcount()
        mask_target = df['rank'] == 0

        # ── Compute profile flags only for targeted rows ───────────────────────
        # (no need to compute for all rows)
        is_shopping_around = df['quote_count'] >= 2
        is_premium = df['total_price'] > HIGH_PRICE
        is_cold_region_heatpump = df['region'].isin(COLD_REGIONS) & df['has_heat_pump']

        # Calculate score (only needed for targeted rows)
        score = (
                is_premium.astype(int) +
                is_cold_region_heatpump.astype(int) -
                df['has_discount'].astype(int) -
                is_shopping_around.astype(int)
        )

        # Determine profile
        price_sensitive = mask_target & (score <= -1)
        value_sensitive = mask_target & (score >= 1)

        # Apply discounts
        extra_pct = pd.Series(0.0, index=df.index)
        extra_pct[price_sensitive] = 0.025  # 2.5% for price sensitive
        extra_pct[value_sensitive] = 0.006  # 0.6% for value sensitive

        extra_amount = df['mt_apres_remise_ht_devis'] * extra_pct

        # Update discount and price
        if 'mt_remise_exceptionnelle_ht' in df.columns:
            df['mt_remise_exceptionnelle_ht'] = df['mt_remise_exceptionnelle_ht'].fillna(0) + extra_amount

        df['mt_apres_remise_ht_devis'] -= extra_amount

        # Debug info
        print(f"\n=== SALES REP SCENARIO ===")
        print(f"Price sensitive customers (2.5% discount): {price_sensitive.sum()}")
        print(f"Value sensitive customers (0.6% discount): {value_sensitive.sum()}")
        print(f"Total discount amount: €{extra_amount.sum():.2f}")

        # Clean up
        df = df.drop(columns=['rank', 'quote_count', 'total_price', 'region',
                              'has_discount', 'has_heat_pump'], errors='ignore')

        return df.sort_index()

def simulate_sales_rep_conversion_shift(df_quotes, model):
    simulator = SalesRepConversionShiftSimulator(df_quotes, model)
    return simulator.run()