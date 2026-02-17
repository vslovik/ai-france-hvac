import pandas as pd


class DiscountSampler:
    def __init__(self, df_sim, random_state=4477):
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

        # 4. Add discount information
        discount_data = []
        for cust_id, group in df_nonconv.groupby('numero_compte'):
            total_price = group['mt_apres_remise_ht_devis'].sum()

            # Calculate current discount
            if 'mt_remise_exceptionnelle_ht' in group.columns:
                current_discount = abs(group['mt_remise_exceptionnelle_ht'].sum())
            else:
                current_discount = 0

            discount_pct = (current_discount / total_price * 100) if total_price > 0 else 0

            discount_data.append({
                'customer_id': cust_id,
                'total_price': total_price,
                'quote_count': len(group),
                'current_discount': current_discount,
                'discount_pct': discount_pct
            })

        df_candidates = pd.DataFrame(discount_data)
        print(f"Non-converted customers: {len(non_converted)}")
        print(f"Candidates with price data: {len(df_candidates)}")

        # 5. Define sampling strategy
        print("\n📊 SAMPLING STRATEGY:")
        print("   1. No existing discount (test introduction)")
        print("   2. Small existing discount (< 2%)")
        print("   3. Medium existing discount (2-5%)")
        print("   4. High price (>€20k)")
        print("   5. Multiple quotes (≥3)")

        # 6. Apply strategic sampling
        import random
        random.seed(self.random_state)

        selected = []

        # 1. No existing discount
        no_discount = df_candidates[df_candidates['current_discount'] == 0]
        if len(no_discount) > 0:
            sample = no_discount.sample(1, random_state=self.random_state)
            selected.append(sample.iloc[0].to_dict())
            print(f"✓ Sampled no-discount: {selected[-1]['customer_id']}")

        # 2. Small existing discount (< 2%)
        small_discount = df_candidates[(df_candidates['discount_pct'] > 0) & (df_candidates['discount_pct'] < 2)]
        if len(small_discount) > 0:
            # Avoid selecting already chosen customer
            available = small_discount[~small_discount['customer_id'].isin([s['customer_id'] for s in selected])]
            if len(available) > 0:
                sample = available.sample(1, random_state=self.random_state)
                selected.append(sample.iloc[0].to_dict())
                print(f"✓ Sampled small-discount: {selected[-1]['customer_id']}")

        # 3. Medium discount (2-5%)
        medium_discount = df_candidates[(df_candidates['discount_pct'] >= 2) & (df_candidates['discount_pct'] <= 5)]
        if len(medium_discount) > 0:
            available = medium_discount[~medium_discount['customer_id'].isin([s['customer_id'] for s in selected])]
            if len(available) > 0:
                sample = available.sample(1, random_state=self.random_state)
                selected.append(sample.iloc[0].to_dict())
                print(f"✓ Sampled medium-discount: {selected[-1]['customer_id']}")

        # 4. High price (> €20,000)
        high_price = df_candidates[df_candidates['total_price'] > 20000]
        if len(high_price) > 0:
            available = high_price[~high_price['customer_id'].isin([s['customer_id'] for s in selected])]
            if len(available) > 0:
                sample = available.sample(1, random_state=self.random_state)
                selected.append(sample.iloc[0].to_dict())
                print(f"✓ Sampled high-price: {selected[-1]['customer_id']}")

        # 5. Multiple quotes (≥ 3 quotes)
        multi_quote = df_candidates[df_candidates['quote_count'] >= 3]
        if len(multi_quote) > 0:
            available = multi_quote[~multi_quote['customer_id'].isin([s['customer_id'] for s in selected])]
            if len(available) > 0:
                sample = available.sample(1, random_state=self.random_state)
                selected.append(sample.iloc[0].to_dict())
                print(f"✓ Sampled multi-quote: {selected[-1]['customer_id']}")

        # Fill remaining with random if needed
        if len(selected) < 5:
            remaining = df_candidates[~df_candidates['customer_id'].isin([s['customer_id'] for s in selected])]
            needed = 5 - len(selected)
            if len(remaining) >= needed:
                additional = remaining.sample(needed, random_state=self.random_state)
                for _, row in additional.iterrows():
                    selected.append(row.to_dict())
                    print(f"✓ Added random: {row['customer_id']}")

        # Convert to DataFrame for clean output
        df_selected = pd.DataFrame(selected)

        print(f"\n🎯 SELECTED {len(df_selected)} DISCOUNT CUSTOMERS:")
        print(df_selected[['customer_id', 'quote_count', 'total_price', 'discount_pct']].to_string(index=False))

        selected_ids = df_selected['customer_id'].tolist()
        print(f"\nSelected IDs: {selected_ids}")

        return selected_ids


def sample_discount_customers(df_sim, random_state=4477):
    sampler = DiscountSampler(df_sim, random_state)
    return sampler.sample()