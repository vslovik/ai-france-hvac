import numpy as np
from ml_inference.inference import safe_predict
from ml_simulation.data import Simulation


class DiscountSimulation(Simulation):

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        super().__init__(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)

    @staticmethod
    def apply_change(df_simulation, cid, family):
        df_quotes = df_simulation[df_simulation['numero_compte'] == cid].copy()
        if len(df_quotes) > 0:
            base_price = df_quotes['mt_apres_remise_ht_devis'].sum()
            discount_amount = base_price * (family / 100)

            # Apply discount to remise column
            if 'mt_remise_exceptionnelle_ht' in df_quotes.columns:
                current_remise = df_quotes['mt_remise_exceptionnelle_ht'].fillna(0)
                df_quotes['mt_remise_exceptionnelle_ht'] = current_remise - discount_amount

            # Update price after discount
            df_quotes['mt_apres_remise_ht_devis'] = base_price - discount_amount

        return df_quotes

    def get_compute_function(self):
        data = self.get_data()

        def compute_func(family=None):
            new_list = []
            new_discounts = []
            for i, cid in enumerate(self.sampled_ids):
                df_quotes = self.df_simulation[self.df_simulation['numero_compte'] == cid].copy()
                new_discounts.append(df_quotes['mt_remise_exceptionnelle_ht'].sum())
                val = data["base"][i]
                if family:
                    if len(df_quotes) > 0:
                        df_quotes_mod = self.apply_change(self.df_simulation, cid, family)
                        new_discounts.append(df_quotes_mod['mt_remise_exceptionnelle_ht'].sum())
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
                'current_discounts': data["current_discounts"],
                'discounts': np.array(new_discounts),
                'delta_avg': np.mean(new_array - data["base"]) if family is not None else 0.0,
            }

        return compute_func


def get_discount_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = DiscountSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()