import numpy as np
from ml_inference.inference import safe_predict
from ml_simulation.data import Simulation


class PriceMatchSimulation(Simulation):

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        super().__init__(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)

    @staticmethod
    def apply_change(df_simulation, cid, family):
        df_quotes = df_simulation[df_simulation['numero_compte'] == cid].copy()
        if len(df_quotes) > 0:
            price = df_quotes['mt_apres_remise_ht_devis'].iloc[0] if 'mt_apres_remise_ht_devis' in df_quotes.columns and len(
                df_quotes) > 0 else 0
            original_price = price
            new_price = original_price * (1 + family)  # reduction is negative
            df_quotes['mt_apres_remise_ht_devis'] = new_price
        return df_quotes

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


def get_price_match_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = PriceMatchSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()