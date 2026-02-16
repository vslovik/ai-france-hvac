import pandas as pd


class PriceMatchSampler:
    def __init__(self, df_sim, random_state=4475):
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

        # 4. Keep customers with at least 1 quote
        df_eligible = df_nonconv[df_nonconv['quote_count'] >= 1].copy()

        # 5. Get price percentiles per product from SIM pool
        product_prices = {}
        for product in df_eligible['famille_equipement_produit'].unique():
            prices = df_eligible[df_eligible['famille_equipement_produit'] == product][
                'mt_apres_remise_ht_devis'].dropna()
            if len(prices) >= 5:  # Need at least 5 for meaningful percentiles
                p30 = prices.quantile(0.3)
                p70 = prices.quantile(0.7)
                product_prices[product] = {'p30': p30, 'p70': p70}

        print(f"Non-converted customers: {len(non_converted)}")
        print(f"Products with price data: {len(product_prices)}")

        # 6. Keep only standard/premium customers (not already budget)
        rows = []
        for cust_id, group in df_eligible.groupby('numero_compte'):
            # Take first quote
            quote = group.iloc[0]
            product = quote['famille_equipement_produit']
            price = quote['mt_apres_remise_ht_devis']

            if product in product_prices and price > product_prices[product]['p30']:
                segment = 'standard' if price <= product_prices[product]['p70'] else 'premium'
                rows.append({
                    'customer_id': cust_id,
                    'product': product,
                    'price': price,
                    'segment': segment
                })

        df_candidates = pd.DataFrame(rows)
        print(f"Standard/premium candidates: {len(df_candidates)}")

        if len(df_candidates) == 0:
            print("⚠️ No standard/premium candidates found! Falling back to random sampling.")
            # Fallback: take any customers with price data
            rows = []
            for cust_id, group in df_eligible.groupby('numero_compte'):
                quote = group.iloc[0]
                if pd.notna(quote['mt_apres_remise_ht_devis']):
                    rows.append({
                        'customer_id': cust_id,
                        'product': quote['famille_equipement_produit'],
                        'price': quote['mt_apres_remise_ht_devis'],
                        'segment': 'unknown'
                    })
            df_candidates = pd.DataFrame(rows)

        # 7. Sample 5 customers with diverse products
        if len(df_candidates) >= 5:
            # Try to get diverse products
            df_diverse = df_candidates.sort_values('customer_id') \
                .drop_duplicates(subset='product', keep='first')

            if len(df_diverse) >= 5:
                sample = df_diverse.sample(n=5, random_state=self.random_state)
            else:
                # Mix of diverse + random
                remaining = df_candidates[~df_candidates['customer_id'].isin(df_diverse['customer_id'])]
                needed = 5 - len(df_diverse)
                extra = remaining.sample(n=needed, random_state=self.random_state)
                sample = pd.concat([df_diverse, extra])
        else:
            # Not enough candidates, take what we have + pad? (should not happen with fallback)
            sample = df_candidates

        print("\n🎯 SELECTED PRICE MATCH CANDIDATES:")
        print(sample[['customer_id', 'product', 'price', 'segment']].to_string(index=False))

        selected_ids = sample['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")
        return selected_ids


def sample_price_match_customers(df_sim, random_state=4475):
    sampler = PriceMatchSampler(df_sim, random_state)
    return sampler.sample()