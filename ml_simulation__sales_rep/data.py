import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

from ml_inference.inference import safe_predict


class SalesRepSimulation:

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
        current_reps = []
        baseline_results = []

        for cust_id in self.sampled_ids:
            quotes = self.df_simulation[self.df_simulation['numero_compte'] == cust_id].copy()
            prob = self.safe_predict(cust_id, quotes, self.pred_model, self.feature_names)
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

            current_rep = quotes['prenom_nom_commercial'].iloc[0] if 'prenom_nom_commercial' in quotes.columns and len(
                quotes) > 0 else 'Unknown'
            current_reps.append(current_rep)

        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])

        # Get predictions for each rep type
        marina_probs = []
        elisabeth_probs = []
        clement_probs = []

        for cust_id in self.sampled_ids:
            quotes = self.df_simulation[self.df_simulation['numero_compte'] == cust_id].copy()

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

        # Determine segments (simplified rule-based for display)
        segments = []
        for i in range(len(self.sampled_ids)):
            if marina_probs[i] - baseline_array[i] > elisabeth_probs[i] - baseline_array[i] + 0.01:
                segments.append('discount_sensitive')
            elif elisabeth_probs[i] - baseline_array[i] > marina_probs[i] - baseline_array[i] + 0.01:
                segments.append('value_sensitive')
            else:
                segments.append('neutral')

        # Color mapping
        segment_colors = {
            'discount_sensitive': '#ff7f0e',  # Orange
            'value_sensitive': '#2ca02c',  # Green
            'neutral': '#1f77b4'  # Blue
        }

        segment_colors_light = {
            'discount_sensitive': '#fdae61',  # Light orange
            'value_sensitive': '#98df8a',  # Light green
            'neutral': '#6baed6'  # Light blue
        }

        color_dark_vals = [segment_colors[seg] for seg in segments]
        color_light_vals = [segment_colors_light[seg] for seg in segments]

        # ─── Compute function ───
        def compute_func(rep_type=None):  # rep_type: None (current), 'marina', 'elisabeth', 'clement'
            model_scen = self.pred_model
            new_list = []

            if rep_type is None:
                new_list = baseline_array.tolist()
            else:
                for i, cid in enumerate(self.sampled_ids):
                    if rep_type == 'marina':
                        new_list.append(marina_probs[i])
                    elif rep_type == 'elisabeth':
                        new_list.append(elisabeth_probs[i])
                    elif rep_type == 'clement':
                        new_list.append(clement_probs[i])
                    else:
                        new_list.append(baseline_array[i])

            new_array = np.array(new_list)
            return {
                'base': baseline_array,
                'new': new_array,
                'regions': regions,
                'products': products,
                'prices': prices,
                'current_reps': current_reps,
                'segments': segments,
                'color_light': color_light_vals,
                'color_dark': color_dark_vals,
                'delta_avg': np.mean(new_array - baseline_array) if rep_type is not None else 0.0
            }

        return compute_func


def get_sales_rep_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    simulation = SalesRepSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return simulation.get_compute_function()