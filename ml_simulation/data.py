import numpy as np
import pandas as pd

from ml_inference.inference import safe_predict
from ml_simulation.constrants import HEAT_PUMP
from ml_simulation.segment import assign_price_value_profile


class Simulation:

    PRODUCT_TIERS = {
        'Chaudière': {'p30': 3646.80, 'p70': 5398.34, 'p90': 7688.06},
        'Poêle': {'p30': 4938.60, 'p70': 6300.31, 'p90': 7573.24},
        'Pompe à chaleur': {'p30': 12546.53, 'p70': 15726.26, 'p90': 18570.97},
        'Climatisation': {'p30': 3816.50, 'p70': 6849.28, 'p90': 11113.26},
        'ECS : Chauffe-eau ou adoucisseur': {'p30': 1380.11, 'p70': 2830.52, 'p90': 3603.93},
        'Photovoltaïque': {'p30': 7565.47, 'p70': 10289.63, 'p90': 13560.08},
        'Autres': {'p30': 1330.80, 'p70': 6100.76, 'p90': 7918.10},
        'Produit VMC': {'p30': 2378.77, 'p70': 5365.29, 'p90': 19652.46},
        'Appareil hybride': {'p30': 14456.04, 'p70': 14773.30, 'p90': 15262.14},
    }

    def __init__(self, pred_model, feature_names, safe_predict, df_simulation, sampled_ids):
        self.pred_model = pred_model
        self.feature_names = feature_names
        self.safe_predict = safe_predict
        self.df_simulation = df_simulation
        self.sampled_ids = sampled_ids
        print(f"Initialized Simulation with {len(sampled_ids)} sampled IDs.")

    @staticmethod
    def apply_change(df_simulation, cid, family):
        return df_simulation[df_simulation['numero_compte'] == cid].copy()  # default: no change

    def get_data(self):
        regions = []
        products = []
        prices = []
        tiers = []
        current_discounts = []
        current_reps = []
        price_value_profiles = []
        baseline_results = []

        # Prepare customer-level data for sampled IDs only
        df_sample = self.df_simulation[self.df_simulation['numero_compte'].isin(self.sampled_ids)].copy()

        # Aggregate once per customer (needed for profile calculation)
        customer_agg = (
            df_sample.groupby('numero_compte', as_index=False)
            .agg(
                quote_count=('numero_compte', 'size'),
                total_price=('mt_apres_remise_ht_devis', 'sum'),
                region=('nom_region', 'first'),
                current_rep=('prenom_nom_commercial', 'first'),
                total_discount=('mt_remise_exceptionnelle_ht', 'sum'),
                has_heat_pump=('famille_equipement_produit', lambda x: (x == HEAT_PUMP).any()),
                main_product=('famille_equipement_produit', 'first'),
                main_price=('mt_apres_remise_ht_devis', 'first')  # representative price
            )
            .rename(columns={'numero_compte': 'customer_id'})
            .assign(
                has_discount=lambda d: d['total_discount'].abs() > 0,
            )
        )

        # Now assign price_value_profile to the aggregated data
        customer_agg = assign_price_value_profile(customer_agg)

        # Create lookup dicts for fast access
        profile_dict = dict(zip(customer_agg['customer_id'], customer_agg['price_value_profile']))
        total_price_dict = dict(zip(customer_agg['customer_id'], customer_agg['total_price']))
        region_dict = dict(zip(customer_agg['customer_id'], customer_agg['region']))
        rep_dict = dict(zip(customer_agg['customer_id'], customer_agg['current_rep']))

        # Build lists using the precomputed values
        for cust_id in self.sampled_ids:
            quotes = df_sample[df_sample['numero_compte'] == cust_id].copy()

            # Baseline prediction
            prob = safe_predict(cust_id, quotes, self.pred_model, self.feature_names)
            baseline_results.append({
                'customer_id': cust_id,
                'baseline_prob': prob
            })

            # Use aggregated values (no more .iloc[0] risk)
            reg = region_dict.get(cust_id, 'Unknown')
            regions.append(reg)

            product = quotes['famille_equipement_produit'].iloc[0] if not quotes.empty else 'Unknown'
            products.append(product)

            price = quotes['mt_apres_remise_ht_devis'].iloc[0] if not quotes.empty else 0
            prices.append(price)

            # Tier logic (using representative price)
            if product in self.PRODUCT_TIERS:
                p30 = self.PRODUCT_TIERS[product]['p30']
                tiers.append('premium' if price > p30 else 'standard')
            else:
                tiers.append('unknown')

            # Discount
            current_disc = abs(
                quotes['mt_remise_exceptionnelle_ht'].sum()) if 'mt_remise_exceptionnelle_ht' in quotes.columns else 0
            current_discounts.append(current_disc)

            # Rep
            current_rep = rep_dict.get(cust_id, 'Unknown')
            current_reps.append(current_rep)

            # Price value profile — now correctly assigned
            profile = profile_dict.get(cust_id, 'Unknown')
            price_value_profiles.append(profile)

        # Final arrays and dicts
        baseline_df = pd.DataFrame(baseline_results)
        baseline_dict = dict(zip(baseline_df['customer_id'], baseline_df['baseline_prob']))
        baseline_array = np.array([baseline_dict[cid] for cid in self.sampled_ids])
        prices_array = np.array(prices)

        return {
            'base': baseline_array,
            'regions': regions,
            'products': products,
            'prices': prices_array,
            'tiers': tiers,
            'current_discounts': np.array(current_discounts),
            'current_reps': current_reps,
            'price_value_profiles': price_value_profiles
        }

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

