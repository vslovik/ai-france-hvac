import pandas as pd


class SalesRepSampler:
    def __init__(self, df_sim, random_state=4478):
        self.df_sim = df_sim
        self.random_state = random_state

    def sample(self):
        # 1. Find non-converted customers
        sim_conv = self.df_sim.groupby('numero_compte')['fg_devis_accepte'].max()
        non_converted = sim_conv[sim_conv == 0].index
        df_nonconv = self.df_sim[self.df_sim['numero_compte'].isin(non_converted)].copy()

        # 2. Add customer-level features
        customer_data = []
        for cust_id, group in df_nonconv.groupby('numero_compte'):
            # Basic info
            quote_count = len(group)
            current_rep = group['prenom_nom_commercial'].iloc[
                0] if 'prenom_nom_commercial' in group.columns else 'Unknown'
            total_price = group['mt_apres_remise_ht_devis'].sum()
            product = group['famille_equipement_produit'].iloc[0]
            region = group['nom_region'].iloc[0] if 'nom_region' in group.columns else 'Unknown'

            # Observable signals for classification
            has_discount = abs(group[
                                   'mt_remise_exceptionnelle_ht'].sum()) > 0 if 'mt_remise_exceptionnelle_ht' in group.columns else False
            is_shopping_around = quote_count >= 2
            is_premium = total_price > 20000  # High price threshold
            is_cold_region_heatpump = (region in ['Normandie', 'Hauts-de-France', 'Grand Est']) and (
                        product == 'Pompe à chaleur')

            # Simple rule-based classification (what sales would do)
            score = 0
            if has_discount:
                score -= 1  # Price-sensitive signal
            if is_shopping_around:
                score -= 1  # Price-sensitive signal
            if is_premium:
                score += 1  # Value-sensitive signal
            if is_cold_region_heatpump:
                score += 1  # Value-sensitive signal (cross-sell opportunity)

            if score >= 2:
                segment = 'value_sensitive'
            elif score <= -1:
                segment = 'price_sensitive'
            else:
                segment = 'neutral'

            customer_data.append({
                'customer_id': cust_id,
                'current_rep': current_rep,
                'quote_count': quote_count,
                'total_price': total_price,
                'product': product,
                'region': region,
                'has_discount': has_discount,
                'is_shopping_around': is_shopping_around,
                'is_premium': is_premium,
                'segment': segment,
                'score': score
            })

        df_candidates = pd.DataFrame(customer_data)
        print(f"\n=== SALES REP SAMPLING ===")
        print(f"Total candidates: {len(df_candidates)}")
        print(f"\n📊 Segment distribution:")
        print(df_candidates['segment'].value_counts().to_string())

        # 3. Sample 5 customers (2 price-sensitive, 2 value-sensitive, 1 neutral)
        import random
        random.seed(self.random_state)

        selected = []

        for segment, n in [('price_sensitive', 2), ('value_sensitive', 2), ('neutral', 1)]:
            segment_cust = df_candidates[df_candidates['segment'] == segment]
            if len(segment_cust) >= n:
                samples = segment_cust.sample(n, random_state=self.random_state)
                for _, row in samples.iterrows():
                    selected.append(row.to_dict())
                    print(f"✓ Sampled {segment}: {row['customer_id']} (score: {row['score']})")

        # Fallback
        if len(selected) < 5:
            remaining = df_candidates[~df_candidates['customer_id'].isin([s['customer_id'] for s in selected])]
            needed = 5 - len(selected)
            if len(remaining) >= needed:
                additional = remaining.sample(needed, random_state=self.random_state)
                for _, row in additional.iterrows():
                    selected.append(row.to_dict())
                    print(f"✓ Added random: {row['customer_id']}")

        df_selected = pd.DataFrame(selected)
        selected_ids = df_selected['customer_id'].tolist()

        print(f"\n🎯 SELECTED {len(selected_ids)} CUSTOMERS:")
        for _, row in df_selected.iterrows():
            print(f"  • {row['customer_id']}: {row['segment']} (rep: {row['current_rep']}, score: {row['score']})")

        return selected_ids


def sample_sales_rep_customers(df_sim, random_state=4478):
    sampler = SalesRepSampler(df_sim, random_state)
    return sampler.sample()