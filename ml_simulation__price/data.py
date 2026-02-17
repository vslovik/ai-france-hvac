import numpy as np
import pandas as pd
from ml_inference.inference import safe_predict


class PriceMatchSimulation:

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        self.pred_model = pred_model
        self.feature_names = feature_names
        self.safe_predict = safe_predict
        self.df_simulation = df_simulation
        self.sampled_ids = sampled_ids

    def get_compute_function(self):
        # Create baseline dataframe and dict
        regions = []
        products = []
        prices = []
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

        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])
        prices_array = np.array(prices)

        # ─── Compute function ───
        def compute_func(family=None):  # reduction: None (actuel) or -0.10, -0.15, -0.20
            model_scen = self.pred_model
            new_list = []

            for i, cid in enumerate(self.sampled_ids):
                df_quotes = self.df_simulation[self.df_simulation['numero_compte'] == cid].copy()

                if family is None or family == 0:
                    new_list.append(baseline_dict[cid])
                else:
                    df_quotes_mod = df_quotes.copy()
                    if len(df_quotes_mod) > 0:
                        # Apply price reduction
                        original_price = prices_array[i]
                        new_price = original_price * (1 + family)  # reduction is negative
                        df_quotes_mod['mt_apres_remise_ht_devis'] = new_price

                    new_val = safe_predict(cid, df_quotes_mod, model_scen, self.feature_names)
                    new_list.append(new_val)

            new_array = np.array(new_list)
            return {
                'base': baseline_array,
                'new': new_array,
                'regions': regions,
                'products': products,
                'prices': prices_array,
                'delta_avg': np.mean(new_array - baseline_array) if family is not None else 0.0
            }

        return compute_func


def get_price_match_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = PriceMatchSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()