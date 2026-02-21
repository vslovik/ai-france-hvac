import numpy as np
import pandas as pd

from ml_simulation.constrants import COLD_REGIONS
from ml_simulation.data import Simulation
from ml_simulation.shift import ConversionShiftSimulator


class CrossSellConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, cross_sell_product='Poêle'):
        super().__init__(df_quotes, model)
        self.cross_sell_product = cross_sell_product

    def apply_change(self) -> pd.DataFrame:
        """
        Applies cross-sell to qualifying customers in a multi-customer DataFrame.

        Qualifying =
          - ≥1 'Pompe à chaleur'
          - 0 'Poêle'
          - nom_region in COLD_REGIONS

        Adds one cross-sell quote per qualifying customer, based on their most recent quote.
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = [
            'numero_compte', 'id_devis', 'mt_apres_remise_ht_devis',
            'dt_creation_devis', 'famille_equipement_produit', 'nom_region'
        ]
        missing = [col for col in required if col not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        if 'COLD_REGIONS' not in globals():
            raise RuntimeError("COLD_REGIONS list not found")

        df = self.df_quotes.copy()
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Qualification per customer
        customer_qual = df.groupby('numero_compte').agg(
            has_heat_pump=('famille_equipement_produit', lambda x: (x == 'Pompe à chaleur').any()),
            has_stove=('famille_equipement_produit', lambda x: (x == 'Poêle').any()),
            region=('nom_region', lambda x: x.mode()[0] if not x.mode().empty else pd.NA),
            most_recent_dt=('dt_creation_devis', 'max'),
            most_recent_idx=('dt_creation_devis', lambda x: x.idxmax())
        ).reset_index()

        qualifying = customer_qual[
            customer_qual['has_heat_pump'] &
            ~customer_qual['has_stove'] &
            customer_qual['region'].isin(COLD_REGIONS)
            ]

        if qualifying.empty:
            return df

        cross_sell_rows = []

        for _, row in qualifying.iterrows():
            most_recent = df.loc[row['most_recent_idx']].copy()

            cross_sell = most_recent.copy()
            cross_sell['famille_equipement_produit'] = self.cross_sell_product

            if self.cross_sell_product in Simulation.PRODUCT_TIERS and 'p30' in Simulation.PRODUCT_TIERS[self.cross_sell_product]:
                cross_sell['mt_apres_remise_ht_devis'] = Simulation.PRODUCT_TIERS[self.cross_sell_product]['p30']
            else:
                cross_sell['mt_apres_remise_ht_devis'] = most_recent['mt_apres_remise_ht_devis'] * 0.70

            # Reset fields
            for col in [
                'mt_marge', 'mt_marge_emis_devis', 'mt_remise_exceptionnelle_ht',
                'mt_ttc_apres_aide_devis', 'mt_ttc_avant_aide_devis',
                'mt_prime_cee', 'mt_prime_maprimerenov',
                'type_equipement_produit', 'marque_produit', 'modele_produit',
                'regroup_famille_equipement_produit'
            ]:
                if col in cross_sell:
                    cross_sell[col] = np.nan

            # Identifiers & descriptive
            cross_sell['id_devis'] = str(most_recent['id_devis']) + '_CS'
            if 'num_devis' in cross_sell:
                cross_sell['num_devis'] = str(most_recent.get('num_devis', '')) + '_CS'
            if 'nom_devis' in cross_sell:
                cross_sell['nom_devis'] = f"{most_recent.get('nom_devis', '')} – Cross-sell {self.cross_sell_product}".strip(
                    ' –')
            if 'type_devis' in cross_sell:
                cross_sell['type_devis'] = 'CROSS_SELL'

            # Clear lifecycle
            for col in [
                'statut_devis', 'fg_devis_emis', 'fg_devis_refuse', 'fg_devis_accepte',
                'dt_signature_devis', 'fg_3_mois_mature', 'dth_emission_devis',
                'dt_emission_calcule_devis', 'lb_statut_preparation_chantier',
                'dt_facture_min', 'dt_facture_max', 'dt_prem_contrat'
            ]:
                if col in cross_sell:
                    cross_sell[col] = pd.NaT if col.startswith(
                        'dt_') else np.nan if 'fg_' in col or 'mt_' in col else None

            cross_sell_rows.append(cross_sell)

        if cross_sell_rows:
            df_updated = pd.concat([df, pd.DataFrame(cross_sell_rows)], ignore_index=True)
            df_updated = df_updated.sort_values(
                ['numero_compte', 'dt_creation_devis', 'id_devis'],
                na_position='last'
            ).reset_index(drop=True)
            return df_updated

        return df


def simulate_cross_sell_conversion_shift(df_quotes, model, cross_sell_product='Poêle'):
    simulator = CrossSellConversionShiftSimulator(df_quotes, model, cross_sell_product)
    return simulator.run()

