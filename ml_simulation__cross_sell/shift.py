import numpy as np
import pandas as pd

from ml_simulation.data import Simulation
from ml_simulation.shift import ConversionShiftSimulator


class CrossSellConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, has_product=None, add_product=None):
        super().__init__(df_quotes, model)
        self.has_product = has_product
        self.add_product = add_product

    def apply_change(self) -> pd.DataFrame:
        """
        Applies cross-sell to qualifying customers:
        - Customers who HAVE has_product
        - Customers who do NOT already have add_product
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        # If no products specified, return original (baseline)
        if self.has_product is None or self.add_product is None:
            return self.df_quotes.copy()

        required = [
            'numero_compte', 'id_devis', 'mt_apres_remise_ht_devis',
            'dt_creation_devis', 'famille_equipement_produit'
        ]
        missing = [col for col in required if col not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        df = self.df_quotes.copy()
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Find most recent quote per customer
        df_sorted = df.sort_values(
            ['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        most_recent = df_sorted.groupby('numero_compte').first().reset_index()

        # Create product boolean matrix
        product_matrix = pd.crosstab(df['numero_compte'], df['famille_equipement_produit']) > 0

        # Check if source product exists in data
        if self.has_product not in product_matrix.columns:
            print(f"Warning: Product '{self.has_product}' not found in data")
            return df

        # Qualifying customers: have source AND don't have target
        has_source = product_matrix[self.has_product]

        if self.add_product in product_matrix.columns:
            has_target = product_matrix[self.add_product]
            qualifying_mask = has_source & ~has_target
        else:
            # Target product doesn't exist yet for anyone
            qualifying_mask = has_source

        qualifying_customers = product_matrix.index[qualifying_mask]

        print(f"\n=== CROSS-SELL: {self.has_product} → {self.add_product} ===")
        print(f"Total customers: {df['numero_compte'].nunique()}")
        print(f"Customers with {self.has_product}: {has_source.sum()}")
        print(
            f"Customers without {self.add_product}: {len(df['numero_compte'].unique()) - (has_target.sum() if self.add_product in product_matrix.columns else 0)}")
        print(f"QUALIFYING: {len(qualifying_customers)} customers")

        if len(qualifying_customers) == 0:
            print("No qualifying customers - returning original data")
            return df

        # Get most recent quotes for qualifying customers
        qualifying_recent = most_recent[most_recent['numero_compte'].isin(qualifying_customers)].copy()

        if qualifying_recent.empty:
            return df

        # Create cross-sell quotes
        cross_sell_rows = qualifying_recent.copy()

        # Update product and date
        cross_sell_rows['famille_equipement_produit'] = self.add_product
        cross_sell_rows['dt_creation_devis'] = pd.to_datetime(cross_sell_rows['dt_creation_devis']) + pd.Timedelta(
            days=7)

        # Set price based on product tier
        if self.add_product in Simulation.PRODUCT_TIERS:
            tier = Simulation.PRODUCT_TIERS[self.add_product]
            # Try p30 first, then p50
            if 'p30' in tier:
                cross_sell_rows['mt_apres_remise_ht_devis'] = tier['p30']
            elif 'p50' in tier:
                cross_sell_rows['mt_apres_remise_ht_devis'] = tier['p50']
            else:
                cross_sell_rows['mt_apres_remise_ht_devis'] = cross_sell_rows['mt_apres_remise_ht_devis'] * 0.7
        else:
            cross_sell_rows['mt_apres_remise_ht_devis'] = cross_sell_rows['mt_apres_remise_ht_devis'] * 0.7

        # Reset product-specific fields
        reset_cols = [
            'mt_marge', 'mt_marge_emis_devis', 'mt_remise_exceptionnelle_ht',
            'mt_ttc_apres_aide_devis', 'mt_ttc_avant_aide_devis',
            'mt_prime_cee', 'mt_prime_maprimerenov',
            'type_equipement_produit', 'marque_produit', 'modele_produit',
            'regroup_famille_equipement_produit'
        ]

        for col in reset_cols:
            if col in cross_sell_rows.columns:
                cross_sell_rows[col] = np.nan

        # Update identifier
        cross_sell_rows['id_devis'] = cross_sell_rows['id_devis'].astype(str) + f"_CS_{self.add_product[:3]}"

        # Clear status fields
        status_cols = [
            'statut_devis', 'fg_devis_emis', 'fg_devis_refuse', 'fg_devis_accepte',
            'dt_signature_devis', 'fg_3_mois_mature', 'dth_emission_devis',
            'dt_emission_calcule_devis', 'lb_statut_preparation_chantier',
            'dt_facture_min', 'dt_facture_max', 'dt_prem_contrat'
        ]

        for col in status_cols:
            if col in cross_sell_rows.columns:
                if col.startswith('dt_'):
                    cross_sell_rows[col] = pd.NaT
                elif col.startswith('fg_') or col.startswith('mt_'):
                    cross_sell_rows[col] = np.nan
                else:
                    cross_sell_rows[col] = None

        print(f"\nAdded {len(cross_sell_rows)} cross-sell quotes")
        print(f"Original quotes: {len(df)}")
        print(f"Final quotes: {len(df) + len(cross_sell_rows)}")

        # Combine and return
        return pd.concat([df, cross_sell_rows], ignore_index=True, sort=False)


def simulate_cross_sell_conversion_shift(df_quotes, model, has_product=None, add_product=None):
    """Run cross-sell simulation for a specific product combination."""
    simulator = CrossSellConversionShiftSimulator(df_quotes, model, has_product, add_product)
    return simulator.run()
