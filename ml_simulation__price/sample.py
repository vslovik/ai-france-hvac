import random

import pandas as pd

from ml_simulation.data import Simulation
from ml_simulation.sample import sample_diverse_products_multi_quote
from ml_simulation.segment import get_nonconverted_customers, find_price_segment_candidates


class PriceMatchSampler:
    def __init__(self, df_sim, random_state=4475):
        self.df_sim = df_sim
        self.random_state = random_state

    def get_eligible_segment(self):
        df_eligible = get_nonconverted_customers(self.df_sim)
        df_candidates = find_price_segment_candidates(
            df_eligible,
            Simulation.PRODUCT_TIERS,
            mode='non_budget'  # → standard + premium (price > p30)
        )
        print(f"Standard/premium candidates: {len(df_candidates)}")
        if len(df_candidates) == 0:
            print("⚠️ No standard/premium candidates → falling back to any valid price")
            df_candidates = find_price_segment_candidates(
                df_eligible,
                Simulation.PRODUCT_TIERS,
                mode='all'
            )
        return df_candidates

    def sample(self):
        random.seed(self.random_state)
        df_candidates = self.get_eligible_segment()
        sample = sample_diverse_products_multi_quote(
            df_candidates, n=5, random_state=self.random_state
        )

        print("\n🎯 SELECTED PRICE MATCH CANDIDATES:")
        print(sample[['customer_id', 'product', 'price', 'segment']].to_string(index=False))

        selected_ids = sample['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")
        return selected_ids


def sample_price_match_customers(df_sim, random_state=4475):
    sampler = PriceMatchSampler(df_sim, random_state)
    return sampler.sample()