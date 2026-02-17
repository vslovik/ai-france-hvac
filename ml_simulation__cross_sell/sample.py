class CrossSellSampler:
    def __init__(self, df_sim, random_state=4479):
        self.df_sim = df_sim
        self.random_state = random_state
        self.heat_pump = 'Pompe à chaleur'
        self.stove = 'Poêle'
        self.cold_regions = ['Normandie', 'Hauts-de-France', 'Grand Est', 'Bourgogne-Franche-Comté']

    def sample(self, n_customers=200, target_sample=5):
        # 1. Find non-converted customers
        sim_conv = self.df_sim.groupby('numero_compte')['fg_devis_accepte'].max()
        non_converted = sim_conv[sim_conv == 0].index
        cust_list = non_converted[:n_customers]

        print(f"\n=== CROSS-SELL SAMPLING: HEAT PUMP → STOVE ===")
        print(f"Scanning {len(cust_list)} customers...")

        # 2. Find eligible customers (NO PREDICTIONS YET)
        eligible = []
        for cust in cust_list:
            quotes = self.df_sim[self.df_sim['numero_compte'] == cust].copy()
            if len(quotes) == 0:
                continue

            products = quotes['famille_equipement_produit'].unique()
            region = quotes['nom_region'].iloc[0] if 'nom_region' in quotes.columns else 'Unknown'

            if self.heat_pump in products and self.stove not in products and region in self.cold_regions:
                eligible.append({
                    'customer_id': cust,
                    'region': region,
                    'quote_count': len(quotes),
                    'quotes': quotes  # Store quotes for later prediction
                })

        print(f"✅ Found {len(eligible)} eligible heat pump owners in cold regions")

        if len(eligible) == 0:
            print("⚠️ No eligible customers found!")
            return []

        # 3. Sample randomly from eligible pool (NO MODEL NEEDED)
        import random
        random.seed(self.random_state)

        target_sample = min(target_sample, len(eligible))
        selected_raw = random.sample(eligible, target_sample)

        # 4. Just return the IDs - predictions happen later in simulation
        selected_ids = [c['customer_id'] for c in selected_raw]

        print(f"\n🎯 SELECTED {len(selected_ids)} CROSS-SELL CANDIDATES:")
        for cust in selected_raw:
            print(f"  • {cust['customer_id']} - {cust['region']}")

        print(f"\nSelected IDs: {selected_ids}")

        return selected_ids


def sample_cross_sell_customers(df_sim, random_state=4479):
    sampler = CrossSellSampler(df_sim, random_state)
    return sampler.sample()