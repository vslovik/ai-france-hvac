import pandas as pd
from ml_simulation.segment import get_nonconverted_customers


class StandardizationSampler:
    def __init__(self, df_sim, price_file, random_state=4477):
        self.df_sim = df_sim
        self.price_file = price_file
        self.random_state = random_state

        # Build set of products that have exact matches in price file
        self.standardized_products = self._build_standardized_product_set()

    def _build_standardized_product_set(self):
        """
        Build set of product keys that have exact matches in price file.
        Key: (famille_equipement_produit, type_equipement_produit, marque_produit)
        """
        standardized = set()

        for _, row in self.price_file.iterrows():
            # Skip if required fields are missing
            if pd.isna(row['famille_equipement_produit']) or pd.isna(row['type_equipement_produit']):
                continue

            key = (
                row['famille_equipement_produit'],
                row['type_equipement_produit'],
                row['marque_produit'] if pd.notna(row['marque_produit']) else ''
            )
            standardized.add(key)

        print(f"✅ Found {len(standardized)} standardized product types")
        return standardized

    def _has_standardized_product(self, customer_id):
        """
        Check if a customer has at least one quote with a standardized product.
        """
        customer_quotes = self.df_sim[self.df_sim['numero_compte'] == customer_id]

        for _, quote in customer_quotes.iterrows():
            product_family = quote.get('famille_equipement_produit', '')
            product_type = quote.get('type_equipement_produit', '')
            product_brand = quote.get('marque_produit', '') if pd.notna(quote.get('marque_produit', '')) else ''

            key = (product_family, product_type, product_brand)

            if key in self.standardized_products:
                return True

        return False

    def get_eligible_segment(self):
        """
        Get non-converted customers who have at least one standardized product.
        """
        # Get all non-converted customers
        df_not_converted = get_nonconverted_customers(self.df_sim)

        # Filter to customers with standardized products
        eligible_customers = []
        product_data = []

        for cust_id, group in df_not_converted.groupby('numero_compte'):
            if self._has_standardized_product(cust_id):
                eligible_customers.append(cust_id)

                # Collect product data for this customer
                for _, quote in group.iterrows():
                    product_family = quote.get('famille_equipement_produit', '')
                    product_type = quote.get('type_equipement_produit', '')
                    product_brand = quote.get('marque_produit', '') if pd.notna(quote.get('marque_produit', '')) else ''
                    key = (product_family, product_type, product_brand)

                    if key in self.standardized_products:
                        # Get standardized price from price file
                        price_row = self.price_file[
                            (self.price_file['famille_equipement_produit'] == product_family) &
                            (self.price_file['type_equipement_produit'] == product_type) &
                            (self.price_file['marque_produit'].fillna('') == product_brand)
                            ].iloc[0]

                        product_data.append({
                            'customer_id': cust_id,
                            'product_family': product_family,
                            'product_type': product_type,
                            'product_brand': product_brand,
                            'original_price': quote['mt_apres_remise_ht_devis'],
                            'std_min_price': price_row['min_pv'],
                            'std_max_price': price_row['max_pv'],
                            'std_avg_price': price_row['moy_pv']
                        })

        # Create candidates DataFrame with customer-level stats
        candidates_list = []

        for cust_id in eligible_customers:
            cust_quotes = df_not_converted[df_not_converted['numero_compte'] == cust_id]
            cust_products = [p for p in product_data if p['customer_id'] == cust_id]

            if len(cust_products) == 0:
                continue

            # Calculate total price change potential
            total_original = sum(p['original_price'] for p in cust_products)
            total_std_min = sum(p['std_min_price'] for p in cust_products)
            total_std_max = sum(p['std_max_price'] for p in cust_products)
            total_std_avg = sum(p['std_avg_price'] for p in cust_products)

            candidate = {
                'customer_id': cust_id,
                'quote_count': len(cust_quotes),
                'standardized_products_count': len(cust_products),
                'total_original_price': total_original,
                'total_std_min_price': total_std_min,
                'total_std_max_price': total_std_max,
                'total_std_avg_price': total_std_avg,
                'potential_savings_min': total_original - total_std_min,
                'potential_savings_max': total_original - total_std_max,
                'potential_savings_avg': total_original - total_std_avg,
                'price_change_pct_min': (
                                                    total_std_min - total_original) / total_original * 100 if total_original > 0 else 0,
                'price_change_pct_max': (
                                                    total_std_max - total_original) / total_original * 100 if total_original > 0 else 0,
                'price_change_pct_avg': (
                                                    total_std_avg - total_original) / total_original * 100 if total_original > 0 else 0
            }

            candidates_list.append(candidate)

        df_candidates = pd.DataFrame(candidates_list)

        print(f"\n📊 ELIGIBLE CUSTOMERS:")
        print(f"   Total non-converted customers: {len(df_not_converted.groupby('numero_compte'))}")
        print(f"   Customers with standardized products: {len(eligible_customers)}")
        if len(eligible_customers) > 0:
            print(
                f"   Percentage: {len(eligible_customers) / len(df_not_converted.groupby('numero_compte')) * 100:.1f}%")

        return df_candidates

    def sample(self):
        """
        Sample 5 customers with standardized products, using diverse criteria.
        """
        df_candidates = self.get_eligible_segment()

        if len(df_candidates) == 0:
            print("⚠️ No eligible customers found!")
            return []

        import random
        random.seed(self.random_state)

        selected = []

        print("\n" + "=" * 80)
        print("SAMPLING STRATEGY")
        print("=" * 80)
        print("   1. Highest price drop (biggest savings)")
        print("   2. Price increase (customer pays more)")
        print("   3. Multiple standardized products")
        print("   4. Small price change (minimal impact)")
        print("   5. Random selection")

        # 1. Highest price drop (biggest savings)
        if len(df_candidates) > 0 and len(selected) < 5:
            highest_drop = df_candidates.nlargest(1, 'potential_savings_avg')
            if len(highest_drop) > 0:
                sample = highest_drop.iloc[0].to_dict()
                selected.append(sample)
                print(f"\n✓ Highest price drop: {sample['customer_id']}")
                print(f"   Savings: €{sample['potential_savings_avg']:,.0f} ({sample['price_change_pct_avg']:.1f}%)")

        # 2. Price increase (customer pays more)
        if len(df_candidates) > 0 and len(selected) < 5:
            price_increase = df_candidates[df_candidates['price_change_pct_avg'] > 0]
            if len(price_increase) > 0:
                price_increase_sorted = price_increase.sort_values('price_change_pct_avg', ascending=True)
                available = price_increase_sorted[
                    ~price_increase_sorted['customer_id'].isin([s['customer_id'] for s in selected])]
                if len(available) > 0:
                    sample = available.iloc[0].to_dict()
                    selected.append(sample)
                    print(f"\n✓ Price increase: {sample['customer_id']}")
                    print(f"   Price change: +{sample['price_change_pct_avg']:.1f}%")

        # 3. Multiple standardized products
        if len(df_candidates) > 0 and len(selected) < 5:
            multi_product = df_candidates[df_candidates['standardized_products_count'] >= 2]
            if len(multi_product) > 0:
                multi_product_sorted = multi_product.sort_values('standardized_products_count', ascending=False)
                available = multi_product_sorted[
                    ~multi_product_sorted['customer_id'].isin([s['customer_id'] for s in selected])]
                if len(available) > 0:
                    sample = available.iloc[0].to_dict()
                    selected.append(sample)
                    print(f"\n✓ Multiple products: {sample['customer_id']}")
                    print(f"   Products: {sample['standardized_products_count']}")

        # 4. Small price change (minimal impact)
        if len(df_candidates) > 0 and len(selected) < 5:
            small_change = df_candidates[abs(df_candidates['price_change_pct_avg']) < 1]
            if len(small_change) > 0:
                small_change_sorted = small_change.sort_values('price_change_pct_avg')
                available = small_change_sorted[
                    ~small_change_sorted['customer_id'].isin([s['customer_id'] for s in selected])]
                if len(available) > 0:
                    sample = available.iloc[0].to_dict()
                    selected.append(sample)
                    print(f"\n✓ Small price change: {sample['customer_id']}")
                    print(f"   Price change: {sample['price_change_pct_avg']:.1f}%")

        # 5. Random selection
        if len(selected) < 5:
            remaining = df_candidates[~df_candidates['customer_id'].isin([s['customer_id'] for s in selected])]
            needed = 5 - len(selected)
            if len(remaining) >= needed:
                additional = remaining.sample(needed, random_state=self.random_state)
                for _, row in additional.iterrows():
                    selected.append(row.to_dict())
                    print(f"\n✓ Random: {row['customer_id']}")
                    print(f"   Price change: {row['price_change_pct_avg']:.1f}%")

        # Convert to DataFrame
        df_selected = pd.DataFrame(selected)

        print("\n" + "=" * 80)
        print("SELECTED CUSTOMERS SUMMARY")
        print("=" * 80)
        display_cols = ['customer_id', 'quote_count', 'standardized_products_count',
                        'total_original_price', 'total_std_avg_price', 'price_change_pct_avg']
        print(df_selected[display_cols].to_string(index=False))

        selected_ids = df_selected['customer_id'].tolist()
        print(f"\n✅ Selected {len(selected_ids)} customers with standardized products")
        print(f"Selected IDs: {selected_ids}")

        return selected_ids


def sample_standardized_customers(df_sim, price_file, random_state=4477):
    """
    Sample customers who have at least one product with standardized pricing.
    """
    sampler = StandardizationSampler(df_sim, price_file, random_state)
    return sampler.sample()