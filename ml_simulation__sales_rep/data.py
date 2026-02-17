import numpy as np

from ml_inference.inference import safe_predict


class SalesRepSimulation:

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        self.pred_model = pred_model
        self.feature_names = feature_names
        self.safe_predict = safe_predict
        self.df_simulation = df_simulation
        self.sampled_ids = sampled_ids

    def get_compute_function(self):
        # Store quotes for on-the-fly calculation
        self.quotes_list = []
        regions = []
        products = []
        prices = []
        current_reps = []

        for cust_id in self.sampled_ids:
            quotes = self.df_simulation[self.df_simulation['numero_compte'] == cust_id].copy()
            self.quotes_list.append(quotes)

            reg = quotes['nom_region'].iloc[0] if 'nom_region' in quotes.columns and len(quotes) > 0 else 'Unknown'
            regions.append(reg)

            product = quotes['famille_equipement_produit'].iloc[
                0] if 'famille_equipement_produit' in quotes.columns and len(quotes) > 0 else 'Unknown'
            products.append(product)

            price = quotes['mt_apres_remise_ht_devis'].iloc[0] if 'mt_apres_remise_ht_devis' in quotes.columns and len(
                quotes) > 0 else 0
            prices.append(price)

            current_rep = quotes['prenom_nom_commercial'].iloc[0] if 'prenom_nom_commercial' in quotes.columns and len(
                quotes) > 0 else 'Unknown'
            current_reps.append(current_rep)

        # Pre-compute segment classification (still needed for logic, but not colors)
        baseline_probs = []
        marina_probs = []
        elisabeth_probs = []
        clement_probs = []

        for i, cust_id in enumerate(self.sampled_ids):
            quotes = self.quotes_list[i].copy()

            # Baseline
            base_prob = self.safe_predict(cust_id, quotes, self.pred_model, self.feature_names)
            baseline_probs.append(base_prob)

            # MARINA (Discount - 2.5%)
            m_quotes = quotes.copy()
            m_price = m_quotes['mt_apres_remise_ht_devis'].sum()
            m_quotes['mt_remise_exceptionnelle_ht'] = -m_price * 0.025
            m_quotes['mt_apres_remise_ht_devis'] = m_price * (1 - 0.025)
            marina_probs.append(self.safe_predict(cust_id, m_quotes, self.pred_model, self.feature_names))

            # ELISABETH (Value - 0.6%)
            e_quotes = quotes.copy()
            e_price = e_quotes['mt_apres_remise_ht_devis'].sum()
            e_quotes['mt_remise_exceptionnelle_ht'] = -e_price * 0.006
            e_quotes['mt_apres_remise_ht_devis'] = e_price * (1 - 0.006)
            elisabeth_probs.append(self.safe_predict(cust_id, e_quotes, self.pred_model, self.feature_names))

            # Clément (Neutral - 1.5%)
            c_quotes = quotes.copy()
            c_price = c_quotes['mt_apres_remise_ht_devis'].sum()
            c_quotes['mt_remise_exceptionnelle_ht'] = -c_price * 0.015
            c_quotes['mt_apres_remise_ht_devis'] = c_price * (1 - 0.015)
            clement_probs.append(self.safe_predict(cust_id, c_quotes, self.pred_model, self.feature_names))

        # Determine segments (for widget to use)
        baseline_array = np.array(baseline_probs)
        segments = []
        for i in range(len(self.sampled_ids)):
            if marina_probs[i] - baseline_array[i] > elisabeth_probs[i] - baseline_array[i] + 0.01:
                segments.append('discount_sensitive')
            elif elisabeth_probs[i] - baseline_array[i] > marina_probs[i] - baseline_array[i] + 0.01:
                segments.append('value_sensitive')
            else:
                segments.append('neutral')

        def compute_func(rep_type=None):
            """Calculate probabilities on-the-fly"""
            baseline_probs = []
            new_probs = []

            for i, cust_id in enumerate(self.sampled_ids):
                quotes = self.quotes_list[i].copy()

                # Calculate baseline (current situation)
                base_prob = self.safe_predict(cust_id, quotes, self.pred_model, self.feature_names)
                baseline_probs.append(base_prob)

                # Calculate new probability based on rep type
                if rep_type is None:
                    new_probs.append(base_prob)
                else:
                    mod_quotes = quotes.copy()

                    if rep_type == 'marina':
                        price = mod_quotes['mt_apres_remise_ht_devis'].sum()
                        mod_quotes['mt_remise_exceptionnelle_ht'] = -price * 0.025
                        mod_quotes['mt_apres_remise_ht_devis'] = price * (1 - 0.025)
                    elif rep_type == 'elisabeth':
                        price = mod_quotes['mt_apres_remise_ht_devis'].sum()
                        mod_quotes['mt_remise_exceptionnelle_ht'] = -price * 0.006
                        mod_quotes['mt_apres_remise_ht_devis'] = price * (1 - 0.006)
                    elif rep_type == 'clement':
                        price = mod_quotes['mt_apres_remise_ht_devis'].sum()
                        mod_quotes['mt_remise_exceptionnelle_ht'] = -price * 0.015
                        mod_quotes['mt_apres_remise_ht_devis'] = price * (1 - 0.015)

                    new_prob = self.safe_predict(cust_id, mod_quotes, self.pred_model, self.feature_names)
                    new_probs.append(new_prob)

            baseline_array = np.array(baseline_probs)
            new_array = np.array(new_probs)

            return {
                'base': baseline_array,
                'new': new_array,
                'regions': regions,
                'products': products,
                'prices': prices,
                'current_reps': current_reps,
                'segments': segments,  # Pass segments to widget
                'delta_avg': np.mean(new_array - baseline_array) if rep_type is not None else 0.0
            }

        return compute_func


def get_sales_rep_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = SalesRepSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()