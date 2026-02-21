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
        For each customer (numero_compte) with EXACTLY ONE quote:
        - Creates one additional (alternative) quote with famille_equipement_produit = new_product
        - Sets mt_apres_remise_ht_devis to PRODUCT_TIERS[new_product]['p30'] if available
        - Falls back to 70% of the original quote price if the product is unknown in PRODUCT_TIERS
        - Resets product-dependent and calculated fields that are no longer valid
        - Generates new identifiers to prevent duplication
        - Clears status, emission, signature, invoice and related fields

        Customers with 0 or ≥2 quotes remain unchanged.

        Requires:
        - PRODUCT_TIERS global dictionary (already defined in your code)
        - Key columns: numero_compte, id_devis, mt_apres_remise_ht_devis, dt_creation_devis
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = ['numero_compte', 'id_devis', 'mt_apres_remise_ht_devis', 'dt_creation_devis']
        missing = [col for col in required if col not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        df = self.df_quotes.copy()
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # ── Find customers with exactly one quote ───────────────────────────────
        quote_counts = df.groupby('numero_compte').size().reset_index(name='quote_count')
        single_quote_customers = quote_counts[quote_counts['quote_count'] == 1]['numero_compte']

        if single_quote_customers.empty:
            return df

        # Extract the single quotes
        mask_single = df['numero_compte'].isin(single_quote_customers)
        originals = df[mask_single].copy()

        # ── Create alternative quotes ───────────────────────────────────────────
        alternatives = originals.copy()
        alternatives['famille_equipement_produit'] = self.new_product

        # Use p30 from PRODUCT_TIERS (preferred business reference price)
        if self.new_product in Simulation.PRODUCT_TIERS and 'p30' in Simulation.PRODUCT_TIERS[self.new_product]:
            alternatives['mt_apres_remise_ht_devis'] = Simulation.PRODUCT_TIERS[self.new_product]['p30']
        else:
            # Fallback consistent with earlier budget logic
            alternatives['mt_apres_remise_ht_devis'] = originals['mt_apres_remise_ht_devis'] * 0.70

        # ── Reset product-specific / calculated fields ──────────────────────────
        reset_cols = [
            'mt_marge', 'mt_marge_emis_devis',
            'mt_remise_exceptionnelle_ht',
            'mt_ttc_apres_aide_devis', 'mt_ttc_avant_aide_devis',
            'mt_prime_cee', 'mt_prime_maprimerenov',
            'type_equipement_produit', 'marque_produit', 'modele_produit',
            'regroup_famille_equipement_produit'
        ]
        for col in reset_cols:
            if col in alternatives.columns:
                alternatives[col] = np.nan

        # ── New identifiers ─────────────────────────────────────────────────────
        alternatives['id_devis'] = alternatives['id_devis'].astype(str) + '_ALT'
        if 'num_devis' in alternatives.columns:
            alternatives['num_devis'] = alternatives['num_devis'].astype(str).replace('nan', '') + '_ALT'

        # Descriptive update (helps readability in reports / UI)
        if 'nom_devis' in alternatives.columns:
            alternatives['nom_devis'] = (
                    alternatives['nom_devis'].astype(str) + f' – Alternative {self.new_product}'
            ).str.strip(' –')

        if 'type_devis' in alternatives.columns:
            alternatives['type_devis'] = 'ALTERNATIVE'

        # ── Clear status / lifecycle fields ─────────────────────────────────────
        clear_cols = [
            'statut_devis', 'fg_devis_emis', 'fg_devis_refuse', 'fg_devis_accepte',
            'dt_signature_devis', 'fg_3_mois_mature',
            'dth_emission_devis', 'dt_emission_calcule_devis',
            'lb_statut_preparation_chantier',
            'dt_facture_min', 'dt_facture_max', 'dt_prem_contrat'
        ]
        for col in clear_cols:
            if col in alternatives.columns:
                if col.startswith('dt_'):
                    alternatives[col] = pd.NaT
                else:
                    alternatives[col] = np.nan if 'fg_' in col or 'mt_' in col else None

        # ── Combine and sort ────────────────────────────────────────────────────
        df_updated = pd.concat([df, alternatives], ignore_index=True)

        df_updated = df_updated.sort_values(
            ['numero_compte', 'dt_creation_devis', 'id_devis'],
            na_position='last'
        ).reset_index(drop=True)

        return df_updated


def simulate_follow_up_conversion_shift(df_quotes, model, new_product='Poêle'):
    simulator = FollowUpConversionShiftSimulator(df_quotes, model, new_product)
    return simulator.run()