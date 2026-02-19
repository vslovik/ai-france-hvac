import random

from ml_simulation.sample import  sample_diverse_products_multi_quote__top_quotes
from ml_simulation.segment import get_mid_range_quote_customers, get_nonconverted_customers


class BudgetAlternativeSampler:
    def __init__(self, df_sim, random_state=4476):
        self.df_sim = df_sim
        self.random_state = random_state

    def get_eligible_segment(self):
        df_eligible = get_nonconverted_customers(self.df_sim)
        df_candidates = get_mid_range_quote_customers(df_eligible)
        return df_candidates

    def sample(self):
        random.seed(self.random_state)
        sample =  sample_diverse_products_multi_quote__top_quotes(self.get_eligible_segment(), n=5, random_state=self.random_state)

        print("\n🎯 SELECTED BUDGET ALTERNATIVE CANDIDATES:")
        print(
            sample[['customer_id', 'product', 'price', 'tier', 'quote_count', 'budget_price', 'savings_pct']].to_string(
                index=False))

        selected_ids = sample['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")
        return selected_ids


def sample_budget_alternative_customers(df_sim, random_state=4476):
    sampler = BudgetAlternativeSampler(df_sim, random_state)
    return sampler.sample()