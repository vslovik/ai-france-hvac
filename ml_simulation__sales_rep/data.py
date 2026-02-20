import numpy as np

from ml_inference.inference import safe_predict
from ml_simulation.data import Simulation


class SalesRepSimulation(Simulation):

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        super().__init__(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)

    def apply_change(self, df_simulation, cid, family):
        df_quotes = df_simulation[df_simulation['numero_compte'] == cid].copy()
        if len(df_quotes) > 0:
            discount = 0.0
            if family == 'marina':
                discount = 0.025
            elif family == 'elisabeth':
                discount = 0.006
            elif family == 'clement':
                discount = 0.015
            price = df_quotes['mt_apres_remise_ht_devis'].sum()
            df_quotes['mt_remise_exceptionnelle_ht'] = -price * discount
            df_quotes['mt_apres_remise_ht_devis'] = price * (1 - discount)
        return df_quotes

    def get_compute_function(self):
        data = self.get_data()

        def compute_func(family=None):
            new_list = []
            for i, cid in enumerate(self.sampled_ids):
                if family is None:
                    val = data["base"][i]
                else:
                    df_quotes_mod = self.apply_change(self.df_simulation, cid, family)
                    val = self.safe_predict(cid, df_quotes_mod, self.pred_model, self.feature_names)
                new_list.append(val)
            new_array = np.array(new_list)
            return {
                'base': data["base"],
                'new': new_array,
                'regions': data["regions"],
                'products': data["products"],
                'prices': data["prices"],
                'tiers': data["tiers"],
                'current_reps': data['current_reps'],
                'segments': data['price_value_profiles'],  # Pass segments to widget
                'delta_avg': np.mean(new_array - data["base"]) if family is not None else 0.0,
            }

        return compute_func


def get_sales_rep_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = SalesRepSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()