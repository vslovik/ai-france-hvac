import numpy as np
import pandas as pd
from ml_inference.inference import safe_predict


class CrossSellSimulation:
    OPTIONS = {
        "Situation actuelle": {
            "family": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Situation actuelle – sans cross-sell"
        },
        "Poêle": {
            "family": "Poêle",
            "emoji": "🔥",
            "color": "#2ca02c",
            "title": "Heat Pump → Poêle (recommandé)"
        },
        "Climatisation": {
            "family": "Climatisation",
            "emoji": "❄️",
            "color": "#ff7f0e",
            "title": "Heat Pump → Climatisation"
        },
        "ECS": {
            "family": "ECS : Chauffe-eau ou adoucisseur",
            "emoji": "💧",
            "color": "#1f77b4",
            "title": "Heat Pump → ECS"
        },
    }

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        self.pred_model = pred_model
        self.feature_names = feature_names
        self.safe_predict = safe_predict
        self.df_simulation = df_simulation
        self.sampled_ids = sampled_ids

    def get_compute_function(self):
        # Store customer data once
        self.customers = []
        for cust_id in self.sampled_ids:
            quotes = self.df_simulation[self.df_simulation['numero_compte'] == cust_id].copy()
            region = quotes['nom_region'].iloc[0] if 'nom_region' in quotes.columns and len(quotes) > 0 else 'Unknown'
            self.customers.append({
                'id': cust_id,
                'region': region,
                'quotes': quotes
            })

        def compute_func(product_family=None):
            baselines = []
            new_probs = []
            regions = []

            for cust in self.customers:
                quotes = cust['quotes'].copy()
                regions.append(cust['region'])

                base_p = self.safe_predict(cust['id'], quotes, self.pred_model, self.feature_names)
                baselines.append(base_p)

                if product_family is None:
                    new_p = base_p
                else:
                    mod = quotes.copy()
                    new_row = mod.iloc[-1:].copy() if len(mod) > 0 else pd.DataFrame(columns=mod.columns)
                    new_row['famille_equipement_produit'] = product_family
                    mod = pd.concat([mod, new_row], ignore_index=True)
                    new_p = self.safe_predict(cust['id'], mod, self.pred_model, self.feature_names)

                new_probs.append(new_p)

            baselines = np.array(baselines)
            new_probs = np.array(new_probs)

            return {
                'baselines': baselines,
                'new_probs': new_probs,
                'regions': regions,
                'delta_avg': np.mean(new_probs - baselines) if product_family else 0.0
            }

        return compute_func


def get_cross_sell_compute_function(pred_model, feature_names, df_simulation, sampled_ids):
    sim = CrossSellSimulation(pred_model, feature_names, safe_predict, df_simulation, sampled_ids)
    return sim.get_compute_function()