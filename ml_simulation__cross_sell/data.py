import numpy as np
import pandas as pd
from ml_inference.inference import safe_predict
from ml_simulation.constrants import COLD_REGIONS
from ml_simulation.data import Simulation


class CrossSellSimulation(Simulation):
    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        super().__init__(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)

    @staticmethod
    def apply_change(df_simulation, cid, family):
        """
        Applies the cross-sell logic to a DataFrame containing quotes from **ONE SINGLE customer** only.

        Qualification criteria:
        - Has at least one 'Pompe à chaleur' quote
        - Has NO 'Poêle' quote
        - Customer is in a cold region (nom_region in COLD_REGIONS)

        If qualified:
        - Creates one additional cross-sell quote based on the most recent existing quote
        - Sets mt_apres_remise_ht_devis = PRODUCT_TIERS[cross_sell_product]['p30'] (if exists)
        - Falls back to 70% of the most recent quote's price otherwise
        - Resets product-dependent & calculated fields
        - Clears status, emission, signature, invoice fields
        - Adds suffix '_CS' to identifiers

        Returns the DataFrame with the new row appended (or unchanged if not qualified).
        """
        df_quotes = df_simulation[df_simulation['numero_compte'] == cid].copy()
        if len(df_quotes) > 0:
            required = [
                'numero_compte', 'id_devis', 'mt_apres_remise_ht_devis',
                'dt_creation_devis', 'famille_equipement_produit', 'nom_region'
            ]
            missing = [col for col in required if col not in df_quotes.columns]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}")

            if 'COLD_REGIONS' not in globals():
                raise RuntimeError("COLD_REGIONS list not found")

            df_quotes['dt_creation_devis'] = pd.to_datetime(df_quotes['dt_creation_devis'], errors='coerce')

            # Qualification checks (using all data present)
            has_heat_pump = (df_quotes['famille_equipement_produit'] == 'Pompe à chaleur').any()
            has_stove = (df_quotes['famille_equipement_produit'] == 'Poêle').any()
            region_values = df_quotes['nom_region'].dropna().unique()
            is_cold = len(region_values) == 1 and region_values[0] in COLD_REGIONS

            if not (has_heat_pump and not has_stove and is_cold):
                return df_quotes  # not qualified

            # Use most recent quote as base
            most_recent = df_quotes.sort_values('dt_creation_devis', ascending=False).iloc[0].copy()

            # Create cross-sell row
            cross_sell = most_recent.copy()
            cross_sell['famille_equipement_produit'] = family

            # Price
            if family in Simulation.PRODUCT_TIERS and 'p30' in Simulation.PRODUCT_TIERS[family]:
                cross_sell['mt_apres_remise_ht_devis'] = Simulation.PRODUCT_TIERS[family]['p30']
            else:
                cross_sell['mt_apres_remise_ht_devis'] = most_recent['mt_apres_remise_ht_devis'] * 0.70

            # Reset product-dependent fields
            reset_cols = [
                'mt_marge', 'mt_marge_emis_devis', 'mt_remise_exceptionnelle_ht',
                'mt_ttc_apres_aide_devis', 'mt_ttc_avant_aide_devis',
                'mt_prime_cee', 'mt_prime_maprimerenov',
                'type_equipement_produit', 'marque_produit', 'modele_produit',
                'regroup_famille_equipement_produit'
            ]
            for col in reset_cols:
                if col in cross_sell:
                    cross_sell[col] = np.nan

            # New identifiers
            cross_sell['id_devis'] = str(most_recent['id_devis']) + '_CS'
            if 'num_devis' in cross_sell:
                cross_sell['num_devis'] = str(most_recent.get('num_devis', '')) + '_CS'

            # Descriptive
            if 'nom_devis' in cross_sell:
                cross_sell[
                    'nom_devis'] = f"{most_recent.get('nom_devis', '')} – Cross-sell {family}".strip(
                    ' –')
            if 'type_devis' in cross_sell:
                cross_sell['type_devis'] = 'CROSS_SELL'

            # Clear lifecycle/status
            clear_cols = [
                'statut_devis', 'fg_devis_emis', 'fg_devis_refuse', 'fg_devis_accepte',
                'dt_signature_devis', 'fg_3_mois_mature', 'dth_emission_devis',
                'dt_emission_calcule_devis', 'lb_statut_preparation_chantier',
                'dt_facture_min', 'dt_facture_max', 'dt_prem_contrat'
            ]
            for col in clear_cols:
                if col in cross_sell:
                    cross_sell[col] = pd.NaT if col.startswith(
                        'dt_') else np.nan if 'fg_' in col or 'mt_' in col else None

            # Append and sort
            df_updated = pd.concat([df_quotes, pd.DataFrame([cross_sell])], ignore_index=True)
            df_updated = df_updated.sort_values(['dt_creation_devis', 'id_devis']).reset_index(drop=True)

            return df_updated

    def get_compute_function(self):
        data = self.get_data()

        def compute_func(family=None):
            new_list = []
            for i, cid in enumerate(self.sampled_ids):
                if family is None:
                    val = data["base"][i]
                else:
                    df_quotes_mod = self.apply_change(self.df_simulation, cid, family)
                    val = safe_predict(cid, df_quotes_mod, self.pred_model, self.feature_names)
                new_list.append(val)
            new_array = np.array(new_list)
            return {
                'base': data["base"],
                'new': new_array,
                'regions': data["regions"],
                'products': data["products"],
                'prices': data["prices"],
                'tiers': data["tiers"],
                'delta_avg': np.mean(new_array - data["base"]) if family is not None else 0.0,
            }

        return compute_func


def get_cross_sell_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    sim = CrossSellSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return sim.get_compute_function()