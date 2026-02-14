import pandas as pd


class FollowUpSampler:
    def __init__(self, df_sim, random_state=4474):
        self.df_sim = df_sim
        self.random_state = random_state

    def sample(self):
        # 1. Find non-converted customers
        sim_conv = self.df_sim.groupby('numero_compte')['fg_devis_accepte'].max()
        non_converted = sim_conv[sim_conv == 0].index

        # 2. Filter the original dataframe once
        df_nonconv = self.df_sim[self.df_sim['numero_compte'].isin(non_converted)].copy()

        # 3. Add quote count per customer
        df_nonconv['quote_count'] = df_nonconv.groupby('numero_compte')['numero_compte'].transform('count')

        # 4. Keep only single-quote non-converted customers
        df_single = df_nonconv[df_nonconv['quote_count'] == 1]

        print(f"Non-converted customers   : {len(non_converted)}")
        print(f"Among them with 1 quote    : {len(df_single['numero_compte'].unique())}")

        # 5. One row per customer (already single quote → one row)
        df_single = df_single[['numero_compte', 'famille_equipement_produit', 'mt_apres_remise_ht_devis']] \
            .rename(columns={
            'numero_compte': 'customer_id',
            'famille_equipement_produit': 'product',
            'mt_apres_remise_ht_devis': 'price'
        })

        # 6. Sample 5 customers with different products when possible
        #    (sort + drop_duplicates gives preference to "first" occurrence per product)
        df_diverse = df_single.sort_values('customer_id') \
            .drop_duplicates(subset='product', keep='first')

        if len(df_diverse) >= 5:
            sample = df_diverse.sample(n=5, random_state=self.random_state)
        else:
            # fill up with random from remaining
            remaining = df_single[~df_single['customer_id'].isin(df_diverse['customer_id'])]
            extra = remaining.sample(n=5 - len(df_diverse), random_state=self.random_state)
            sample = pd.concat([df_diverse, extra])

        print(sample)
        selected_ids = [row['customer_id'] for i, row in sample.iterrows()]
        print(selected_ids)
        return selected_ids


def sample_follow_up_customers(df_sim, random_state=4474):
    sampler = FollowUpSampler(df_sim, random_state)
    return sampler.sample()