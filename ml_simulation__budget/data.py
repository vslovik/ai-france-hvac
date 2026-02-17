import numpy as np
import pandas as pd
from ml_inference.inference import safe_predict


class BudgetAlternativeSimulation:

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
        tiers = []
        baseline_results = []

        # First, get price tiers for each product (needed for budget calculation)
        df_all = self.df_simulation.copy()
        product_tiers = {}
        for product in df_all['famille_equipement_produit'].unique():
            prices_prod = df_all[df_all['famille_equipement_produit'] == product]['mt_apres_remise_ht_devis'].dropna()
            if len(prices_prod) >= 5:
                p30 = prices_prod.quantile(0.3)
                product_tiers[product] = p30

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

            # Determine tier (premium if price > p70, standard otherwise)
            if product in product_tiers:
                tiers.append('premium' if price > product_tiers[product] else 'standard')
            else:
                tiers.append('unknown')

        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])
        prices_array = np.array(prices)

        # ─── Compute function ───
        def compute_func(scenario=None):  # scenario: None (actuel) or 'budget'
            model_scen = self.pred_model
            new_list = []
            budget_prices = []

            for i, cid in enumerate(self.sampled_ids):
                df_quotes = self.df_simulation[self.df_simulation['numero_compte'] == cid].copy()
                product = products[i]

                if scenario is None:
                    new_list.append(baseline_dict[cid])
                    budget_prices.append(prices_array[i])
                else:
                    df_quotes_mod = df_quotes.copy()
                    if len(df_quotes_mod) > 0 and product in product_tiers:
                        # Apply budget price (30th percentile)
                        budget_price = product_tiers[product]
                        df_quotes_mod['mt_apres_remise_ht_devis'] = budget_price
                        budget_prices.append(budget_price)
                    else:
                        budget_prices.append(prices_array[i])

                    new_val = safe_predict(cid, df_quotes_mod, model_scen, self.feature_names)
                    new_list.append(new_val)

            new_array = np.array(new_list)
            return {
                'base': baseline_array,
                'new': new_array,
                'regions': regions,
                'products': products,
                'prices': prices_array,
                'budget_prices': np.array(budget_prices),
                'tiers': tiers,
                'delta_avg': np.mean(new_array - baseline_array) if scenario is not None else 0.0
            }

        return compute_func


def get_budget_alternative_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = BudgetAlternativeSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()