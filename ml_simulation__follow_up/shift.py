import numpy as np
import pandas as pd

from ml_simulation.data import Simulation
from ml_simulation.shift import ConversionShiftSimulator


class FollowUpConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, new_product='Poêle'):
        super().__init__(df_quotes, model)
        self.new_product = new_product

    def apply_change(self) -> pd.DataFrame:
        """
        For each customer with EXACTLY ONE quote:
        - Creates one additional (alternative) quote with famille_equipement_produit = new_product
        - Sets date to original date + 7 days
        - Sets price to PRODUCT_TIERS[new_product]['p30'] if available
        - Falls back to 70% of original price if product unknown
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = ['numero_compte', 'id_devis', 'mt_apres_remise_ht_devis', 'dt_creation_devis']
        missing = [col for col in required if col not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        df = self.df_quotes.copy()
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Find customers with exactly one quote
        quote_counts = df.groupby('numero_compte').size().reset_index(name='quote_count')
        single_quote_customers = quote_counts[quote_counts['quote_count'] == 1]['numero_compte']

        print(f"\n=== FOLLOW-UP SIMULATION: {self.new_product} ===")
        print(f"Total customers: {df['numero_compte'].nunique()}")
        print(f"Single-quote customers: {len(single_quote_customers)}")

        if single_quote_customers.empty:
            print("No single-quote customers found - returning original data")
            return df

        # Extract the single quotes
        mask_single = df['numero_compte'].isin(single_quote_customers)
        originals = df[mask_single].copy()

        print(f"Creating {len(originals)} alternative quotes")

        # Create alternative quotes
        alternatives = originals.copy()
        alternatives['famille_equipement_produit'] = self.new_product
        alternatives['dt_creation_devis'] = originals['dt_creation_devis'] + pd.Timedelta(days=7)

        # Set price
        if self.new_product in Simulation.PRODUCT_TIERS:
            tier = Simulation.PRODUCT_TIERS[self.new_product]
            price = tier.get('p30') or tier.get('p50')
            if price is not None:
                alternatives['mt_apres_remise_ht_devis'] = price
                print(f"  Using tiered pricing: {price:.0f}")
            else:
                alternatives['mt_apres_remise_ht_devis'] = originals['mt_apres_remise_ht_devis'] * 0.7
        else:
            alternatives['mt_apres_remise_ht_devis'] = originals['mt_apres_remise_ht_devis'] * 0.7
            print(f"  Using fallback pricing (70% of original)")

        # Reset fields
        reset_cols = [
            'mt_marge', 'mt_marge_emis_devis', 'mt_remise_exceptionnelle_ht',
            'mt_ttc_apres_aide_devis', 'mt_ttc_avant_aide_devis',
            'mt_prime_cee', 'mt_prime_maprimerenov',
            'type_equipement_produit', 'marque_produit', 'modele_produit',
            'regroup_famille_equipement_produit'
        ]
        for col in reset_cols:
            if col in alternatives.columns:
                alternatives[col] = np.nan

        # New identifiers
        alternatives['id_devis'] = alternatives['id_devis'].astype(str) + f'_ALT_{self.new_product[:3]}'

        # Clear status fields
        status_cols = [
            'statut_devis', 'fg_devis_emis', 'fg_devis_refuse', 'fg_devis_accepte',
            'dt_signature_devis', 'fg_3_mois_mature', 'dth_emission_devis',
            'dt_emission_calcule_devis', 'lb_statut_preparation_chantier',
            'dt_facture_min', 'dt_facture_max', 'dt_prem_contrat'
        ]
        for col in status_cols:
            if col in alternatives.columns:
                if col.startswith('dt_'):
                    alternatives[col] = pd.NaT
                else:
                    alternatives[col] = np.nan

        # Combine and return
        df_updated = pd.concat([df, alternatives], ignore_index=True)

        print(f"\nAdded {len(alternatives)} follow-up quotes")
        print(f"Original quotes: {len(df)}")
        print(f"Final quotes: {len(df_updated)}")

        return df_updated


def simulate_follow_up_conversion_shift(df_quotes, model, new_product='Poêle'):
    """Simulate offering an alternative product to single-quote customers, 7 days later."""
    simulator = FollowUpConversionShiftSimulator(df_quotes, model, new_product)
    return simulator.run()
