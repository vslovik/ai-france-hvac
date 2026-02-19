import random

from ml_simulation.segment import get_customers_with_heat_pump_quote_with_no_stove_quote_in_cold_region


class CrossSellSampler:
    def __init__(self, df_sim, random_state=4479):
        self.df_sim = df_sim
        self.random_state = random_state

    def get_eligible_segment(self):
        df_candidates = get_customers_with_heat_pump_quote_with_no_stove_quote_in_cold_region(self.df_sim)
        return df_candidates

    def sample(self):
        random.seed(self.random_state)
        eligible = self.get_eligible_segment()
        sample = eligible.sample(n=5, random_state=self.random_state)
        selected_ids = sample['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")
        return selected_ids


def sample_cross_sell_customers(df_sim, random_state=4479):
    sampler = CrossSellSampler(df_sim, random_state)
    return sampler.sample()