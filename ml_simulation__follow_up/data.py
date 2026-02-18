import numpy as np
import pandas as pd

from ml_inference.inference import safe_predict
from ml_simulation.data import Simulation


class FollowUpSimulation(Simulation):

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        super().__init__(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)

    @staticmethod
    def apply_change(df_simulation, cid, family):
        df_quotes = df_simulation[df_simulation['numero_compte'] == cid].copy()
        if len(df_quotes) > 0:
            new_row = df_quotes.iloc[-1:].copy()
            new_row['famille_equipement_produit'] = family
            df_quotes = pd.concat([df_quotes, new_row], ignore_index=True)
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


def get_simulation_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = FollowUpSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()
