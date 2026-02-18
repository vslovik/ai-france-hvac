import numpy as np
from ml_inference.inference import safe_predict
from ml_simulation.data import Simulation


class BudgetSimulation(Simulation):

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        super().__init__(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)

    @staticmethod
    def apply_change(df_simulation, cid, family):
        """
        Switch to budget version with capped reduction using hardcoded tiers
        """
        df_quotes = df_simulation[df_simulation['numero_compte'] == cid].copy()
        price = df_quotes['mt_apres_remise_ht_devis'].sum()
        product = df_quotes['famille_equipement_produit'].iloc[0]

        if product in Simulation.PRODUCT_TIERS:
            new_price = Simulation.PRODUCT_TIERS[product]['p30']
        else:
            new_price = price * 0.7

        # Cap the reduction
        max_reduction = 0.3
        reduction = 1 - (new_price / price)
        if reduction > max_reduction:
            new_price = price * (1 - max_reduction)
        else:
            new_price = new_price

        df_quotes['mt_apres_remise_ht_devis'] = new_price
        return df_quotes

    def get_compute_function(self):
        data = self.get_data()

        def compute_func(family=None):  # scenario: None (actuel) or 'budget'
            new_list = []
            budget_prices = []
            for i, cid in enumerate(self.sampled_ids):
                df_quotes = self.df_simulation[self.df_simulation['numero_compte'] == cid].copy()
                val = data["base"][i]
                budget_price = df_quotes['mt_apres_remise_ht_devis'].sum()
                if family is not None:
                    df_quotes_mod = df_quotes.copy()
                    if len(df_quotes_mod) > 0:
                        # Apply budget price (30th percentile)
                        df_quotes_mod = self.apply_change(self.df_simulation, cid, family)
                        budget_price = df_quotes_mod['mt_apres_remise_ht_devis'].sum()
                        val = safe_predict(cid, df_quotes_mod, self.pred_model, self.feature_names)

                budget_prices.append(budget_price)
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
                'budget_prices': np.array(budget_prices),
            }

        return compute_func


def get_budget_alternative_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = BudgetSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()