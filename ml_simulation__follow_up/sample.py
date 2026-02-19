import random
from ml_simulation.sample import sample_for_single_quote_customers
from ml_simulation.segment import get_nonconverted_customers, get_single_quote_customers


class FollowUpSampler:
    def __init__(self, df_sim, random_state=4474):
        self.df_sim = df_sim
        self.random_state = random_state

    def get_eligible_segment(self):
        return get_single_quote_customers(self.df_sim)

    def sample(self):
        random.seed(self.random_state)
        sample = sample_for_single_quote_customers(self.get_eligible_segment(), n=5, random_state=self.random_state)
        print("\n🎯 SELECTED FOLLOW UP CANDIDATES:")
        print(
            sample[['customer_id', 'product', 'price', 'quote_count']].to_string(
                index=False))

        selected_ids = sample['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")
        return selected_ids


def sample_follow_up_customers(df_sim, random_state=4474):
    sampler = FollowUpSampler(df_sim, random_state)
    return sampler.sample()