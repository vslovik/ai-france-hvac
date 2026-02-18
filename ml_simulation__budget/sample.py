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

        # ===== CORRECTED TARGETING =====
        # Target MID-RANGE customers (30th-70th percentile)
        # These are the ones who might consider a budget option
        # Premium customers (>p70) should NOT be targeted for budget

        rows = []
        for cust_id, group in df_eligible.groupby('numero_compte'):
            # Take first quote
            quote = group.iloc[0]
            product = quote['famille_equipement_produit']
            price = quote['mt_apres_remise_ht_devis']
            quote_count = len(group)

            if product in product_prices:
                p30 = product_prices[product]['p30']
                p70 = product_prices[product]['p70']

                # Target MID-RANGE customers (price between p30 and p70)
                if p30 <= price <= p70:
                    # Also prefer customers who are shopping around (multiple quotes)
                    # They're more price-sensitive
                    rows.append({
                        'customer_id': cust_id,
                        'product': product,
                        'price': price,
                        'tier': 'mid_range',
                        'quote_count': quote_count,
                        'budget_price': p30,  # Target budget price (p30)
                        'savings': price - p30,
                        'savings_pct': (price - p30) / price * 100
                    })

        df_candidates = pd.DataFrame(rows)
        print(f"\n🎯 MID-RANGE candidates (p30-p70): {len(df_candidates)}")

        # If not enough mid-range customers, broaden to include
        # customers just above p70 who might still consider budget
        if len(df_candidates) < 10:
            print("⚠️ Few mid-range candidates, broadening to include lower-premium...")
            for cust_id, group in df_eligible.groupby('numero_compte'):
                quote = group.iloc[0]
                product = quote['famille_equipement_produit']
                price = quote['mt_apres_remise_ht_devis']
                quote_count = len(group)

                if product in product_prices:
                    p70 = product_prices[product]['p70']
                    p85 = product_prices[product]['p70'] * 1.15  # Approximate 85th percentile

                    # Include customers just above p70 (still potentially interested in savings)
                    if p70 < price <= p85:
                        rows.append({
                            'customer_id': cust_id,
                            'product': product,
                            'price': price,
                            'tier': 'premium_light',
                            'quote_count': quote_count,
                            'budget_price': product_prices[product]['p30'],
                            'savings': price - product_prices[product]['p30'],
                            'savings_pct': (price - product_prices[product]['p30']) / price * 100
                        })
            df_candidates = pd.DataFrame(rows)
            print(f"  → Now {len(df_candidates)} candidates including premium-light")

        # 7. Sample 5 customers with diverse products
        if len(df_candidates) >= 5:
            # Prioritize customers with multiple quotes (more price-sensitive)
            df_candidates = df_candidates.sort_values('quote_count', ascending=False)

            # Try to get diverse products
            df_diverse = df_candidates.drop_duplicates(subset='product', keep='first')

            if len(df_diverse) >= 5:
                sample = df_diverse.head(5)
            else:
                # Mix of diverse + highest quote count
                remaining = df_candidates[~df_candidates['customer_id'].isin(df_diverse['customer_id'])]
                needed = 5 - len(df_diverse)
                remaining = remaining.sort_values('quote_count', ascending=False)
                extra = remaining.head(needed)
                sample = pd.concat([df_diverse, extra])
        else:
            # Not enough candidates, take what we have
            sample = df_candidates

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