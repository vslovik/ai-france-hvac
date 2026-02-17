import pandas as pd


class BudgetAlternativeSampler:
    def __init__(self, df_sim, random_state=4476):
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
        price_stats = []

        for product in df_eligible['famille_equipement_produit'].unique():
            prices = df_eligible[df_eligible['famille_equipement_produit'] == product][
                'mt_apres_remise_ht_devis'].dropna()
            if len(prices) >= 5:
                p30 = prices.quantile(0.3)
                p70 = prices.quantile(0.7)
                p90 = prices.quantile(0.9)
                product_prices[product] = {'p30': p30, 'p70': p70, 'p90': p90}
                price_stats.append({
                    'product': product,
                    'count': len(prices),
                    'p30': p30,
                    'p70': p70,
                    'p90': p90
                })

        print(f"Non-converted customers: {len(non_converted)}")
        print(f"Products with price data: {len(product_prices)}")

        # Show price stats
        df_stats = pd.DataFrame(price_stats).sort_values('count', ascending=False)
        print("\n📊 PRODUCT PRICE TIERS (from simulation pool):")
        print(df_stats.to_string(index=False))

        # 6. Target customers with PREMIUM products (top 30% price)
        rows = []
        for cust_id, group in df_eligible.groupby('numero_compte'):
            # Take first quote
            quote = group.iloc[0]
            product = quote['famille_equipement_produit']
            price = quote['mt_apres_remise_ht_devis']

            if product in product_prices and price >= product_prices[product]['p70']:  # Premium tier
                rows.append({
                    'customer_id': cust_id,
                    'product': product,
                    'price': price,
                    'tier': 'premium',
                    'budget_price': product_prices[product]['p30']  # Target budget price
                })

        df_candidates = pd.DataFrame(rows)
        print(f"\nPremium product candidates: {len(df_candidates)}")

        if len(df_candidates) == 0:
            print("⚠️ No premium candidates found! Falling back to standard pricing.")
            # Fallback: take standard customers
            for cust_id, group in df_eligible.groupby('numero_compte'):
                quote = group.iloc[0]
                product = quote['famille_equipement_produit']
                price = quote['mt_apres_remise_ht_devis']
                if product in product_prices and price >= product_prices[product]['p30']:
                    rows.append({
                        'customer_id': cust_id,
                        'product': product,
                        'price': price,
                        'tier': 'standard',
                        'budget_price': product_prices[product]['p30']
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
            # Not enough candidates, take what we have
            sample = df_candidates

        print("\n🎯 SELECTED BUDGET ALTERNATIVE CANDIDATES:")
        print(sample[['customer_id', 'product', 'price', 'tier', 'budget_price']].to_string(index=False))

        selected_ids = sample['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")
        return selected_ids


def sample_budget_alternative_customers(df_sim, random_state=4476):
    sampler = BudgetAlternativeSampler(df_sim, random_state)
    return sampler.sample()