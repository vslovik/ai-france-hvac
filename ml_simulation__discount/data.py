import numpy as np
import pandas as pd
from ml_inference.inference import safe_predict


class DiscountSimulation:

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

            # Get current discount
            if 'mt_remise_exceptionnelle_ht' in quotes.columns:
                current_disc = abs(quotes['mt_remise_exceptionnelle_ht'].sum())
            else:
                current_disc = 0
            current_discounts.append(current_disc)

        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])
        prices_array = np.array(prices)
        discounts_array = np.array(current_discounts)

        # ─── Compute function ───
        def compute_func(discount_percent=None):  # discount_percent: None (actuel) or 1.0, 1.5, 2.0, etc.
            model_scen = self.pred_model
            new_list = []
            new_discounts = []

            for i, cid in enumerate(self.sampled_ids):
                df_quotes = self.df_simulation[self.df_simulation['numero_compte'] == cid].copy()

                if discount_percent is None or discount_percent == 0:
                    new_list.append(baseline_dict[cid])
                    new_discounts.append(discounts_array[i])
                else:
                    df_quotes_mod = df_quotes.copy()
                    if len(df_quotes_mod) > 0:
                        # Apply strategic discount
                        base_price = prices_array[i]
                        discount_amount = base_price * (discount_percent / 100)

                        # Apply discount to remise column
                        if 'mt_remise_exceptionnelle_ht' in df_quotes_mod.columns:
                            current_remise = df_quotes_mod['mt_remise_exceptionnelle_ht'].fillna(0)
                            df_quotes_mod['mt_remise_exceptionnelle_ht'] = current_remise - discount_amount
                            new_discounts.append(discounts_array[i] + discount_amount)
                        else:
                            new_discounts.append(discounts_array[i])

                        # Update price after discount
                        df_quotes_mod['mt_apres_remise_ht_devis'] = base_price - discount_amount

                    new_val = safe_predict(cid, df_quotes_mod, model_scen, self.feature_names)
                    new_list.append(new_val)

            new_array = np.array(new_list)
            return {
                'base': baseline_array,
                'new': new_array,
                'regions': regions,
                'products': products,
                'prices': prices_array,
                'discounts': np.array(new_discounts),
                'current_discounts': discounts_array,
                'delta_avg': np.mean(new_array - baseline_array) if discount_percent is not None else 0.0
            }

        return compute_func


def get_discount_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = DiscountSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()