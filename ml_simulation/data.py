import numpy as np
import pandas as pd

from ml_inference.inference import safe_predict


class Simulation:

    PRODUCT_TIERS = {
        'Chaudière': {'p30': 3646.80, 'p70': 5398.34, 'p90': 7688.06},
        'Poêle': {'p30': 4938.60, 'p70': 6300.31, 'p90': 7573.24},
        'Pompe à chaleur': {'p30': 12546.53, 'p70': 15726.26, 'p90': 18570.97},
        'Climatisation': {'p30': 3816.50, 'p70': 6849.28, 'p90': 11113.26},
        'ECS : Chauffe-eau ou adoucisseur': {'p30': 1380.11, 'p70': 2830.52, 'p90': 3603.93},
        'Photovoltaïque': {'p30': 7565.47, 'p70': 10289.63, 'p90': 13560.08},
        'Autres': {'p30': 1330.80, 'p70': 6100.76, 'p90': 7918.10},
        'Produit VMC': {'p30': 2378.77, 'p70': 5365.29, 'p90': 19652.46},
        'Appareil hybride': {'p30': 14456.04, 'p70': 14773.30, 'p90': 15262.14},
    }

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        self.pred_model = pred_model
        self.feature_names = feature_names
        self.safe_predict = safe_predict
        self.df_simulation = df_simulation
        self.sampled_ids = sampled_ids
        print(f"Initialized Simulation with {len(sampled_ids)} sampled IDs.")

    def get_data(self):
        regions = []
        products = []
        prices = []
        tiers = []
        current_discounts = []
        baseline_results = []
        for cust_id in self.sampled_ids:

            quotes = self.df_simulation[self.df_simulation['numero_compte'] == cust_id].copy()
            prob = safe_predict(cust_id, quotes, self.pred_model, self.feature_names)
            baseline_results.append({
                'customer_id': cust_id,
                'baseline_prob': prob
            })

            reg = quotes['nom_region'].iloc[0] if 'nom_region' in quotes.columns and len(quotes) > 0 else 'Unknown'
            regions.append(reg)
            product = quotes['famille_equipement_produit'].iloc[
                0] if 'famille_equipement_produit' in quotes.columns and len(quotes) > 0 else 'Unknown'
            products.append(product)
            price = quotes['mt_apres_remise_ht_devis'].iloc[0] if 'mt_apres_remise_ht_devis' in quotes.columns and len(
                quotes) > 0 else 0
            prices.append(price)
            if product in self.PRODUCT_TIERS:
                tiers.append('premium' if price > self.PRODUCT_TIERS[product]['p30'] else 'standard')
            else:
                tiers.append('unknown')
            if 'mt_remise_exceptionnelle_ht' in quotes.columns:
                current_disc = abs(quotes['mt_remise_exceptionnelle_ht'].sum())
            else:
                current_disc = 0
            current_discounts.append(current_disc)

        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])
        prices_array = np.array(prices)

        return {
            'base': baseline_array,  # fixed reference array
            'regions': regions,
            'products': products,
            'prices': prices_array,
            'tiers': tiers,
            'current_discounts': np.array(current_discounts)
        }

    def get_compute_function(self):
        data = self.get_data()

        def compute_func(family=None):
            new_list = []
            for i, cid in enumerate(self.sampled_ids):
                if family is None or family == 0:
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
                'delta_avg': np.mean(new_array - data["base"]) if family is not None else 0.0
            }

        return compute_func

