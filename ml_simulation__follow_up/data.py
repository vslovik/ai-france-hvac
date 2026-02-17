import numpy as np
import pandas as pd

from ml_inference.inference import safe_predict


class FollowUpSimulation:

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        self.pred_model = pred_model
        self.feature_names = feature_names
        self.safe_predict = safe_predict
        self.df_simulation = df_simulation
        self.sampled_ids = sampled_ids

    def get_compute_function(self):
        regions = []
        products = []
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

        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])

        def compute_func(family=None):
            model_scen = self.pred_model  # ← replace with real model if dropout affects it

            new_list = []

            for cid in self.sampled_ids:
                df_quotes = self.df_simulation[self.df_simulation['numero_compte'] == cid].copy()

                if family is None:
                    new_list.append(baseline_dict[cid])
                else:
                    df_quotes_mod = df_quotes.copy()
                    if len(df_quotes_mod) > 0:
                        new_row = df_quotes_mod.iloc[-1:].copy()
                        new_row['famille_equipement_produit'] = family
                        df_quotes_mod = pd.concat([df_quotes_mod, new_row], ignore_index=True)
                    new_val = safe_predict(cid, df_quotes_mod, model_scen, self.feature_names)
                    new_list.append(new_val)

            new_array = np.array(new_list)
            return {
                'base': baseline_array,  # fixed reference array
                'new': new_array,
                'regions': regions,
                'families': products,
                'delta_avg': np.mean(new_array - baseline_array) if family else 0.0
            }

        return compute_func


def get_simulation_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = FollowUpSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()
