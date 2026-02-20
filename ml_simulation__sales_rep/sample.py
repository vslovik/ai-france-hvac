import random

import pandas as pd

from ml_simulation.constrants import COLD_REGIONS, HEAT_PUMP, HIGH_PRICE
from ml_simulation.segment import get_nonconverted_customers, assign_price_value_profile


class SalesRepSampler:
    def __init__(self, df_sim, random_state=4478):
        self.df_sim = df_sim
        self.random_state = random_state

    def get_eligible_segment(self):
        df = (
            get_nonconverted_customers(self.df_sim)
            .groupby('numero_compte', as_index=False)
            .agg(
                quote_count=('numero_compte', 'size'),
                current_rep=('prenom_nom_commercial', 'first'),
                total_price=('mt_apres_remise_ht_devis', 'sum'),
                product=('famille_equipement_produit', 'first'),
                region=('nom_region', 'first'),
                has_discount=('mt_remise_exceptionnelle_ht', lambda x: x.abs().sum() > 0),
                has_heat_pump=('famille_equipement_produit', lambda x: (x == HEAT_PUMP).any()),
            )
            .rename(columns={'numero_compte': 'customer_id'})
            .pipe(assign_price_value_profile)
        )

        if df.empty:
            print("⚠️ No eligible customers")
        else:
            print(f"Found {len(df):,} candidates • {df['price_value_profile'].value_counts().to_dict()}")

        return df

    def sample(self):
        df_candidates = self.get_eligible_segment()
        if df_candidates.empty:
            print("⚠️ No candidates available.")
            return []

        targets = [('price_sensitive', 2), ('value_sensitive', 2), ('neutral', 1)]
        selected = []
        for segment, n in targets:
            seg_df = df_candidates[df_candidates['price_value_profile'] == segment]
            if len(seg_df) == 0:
                continue

            take = min(n, len(seg_df))
            sample_df = seg_df.sample(n=take, random_state=self.random_state)

            for _, row in sample_df.iterrows():
                d = row.to_dict()
                selected.append(d)
                print(f"✓ {segment}: {d['customer_id']} (score: {d['score']})")

        # Fill up to 5 if needed
        if len(selected) < 5:
            taken_ids = {d['customer_id'] for d in selected}
            remaining = df_candidates[~df_candidates['customer_id'].isin(taken_ids)]
            needed = 5 - len(selected)

            if len(remaining) >= needed:
                extra = remaining.sample(n=needed, random_state=self.random_state)
                for _, row in extra.iterrows():
                    d = row.to_dict()
                    selected.append(d)
                    print(f"✓ fallback: {d['customer_id']} ({d['price_value_profile']}, score: {d['score']})")

        if not selected:
            return []

        df_selected = pd.DataFrame(selected)
        selected_ids = df_selected['customer_id'].tolist()

        print(f"\n🎯 Selected {len(selected_ids)} customers:")
        for d in selected:
            print(f" • {d['customer_id']}: {d['price_value_profile']} (rep: {d['current_rep']}, score: {d['score']})")

        return selected_ids


def sample_sales_rep_customers(df_sim, random_state=4478):
    sampler = SalesRepSampler(df_sim, random_state)
    return sampler.sample()