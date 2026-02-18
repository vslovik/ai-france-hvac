def calculate_product_tiers(df, min_samples=20):
    """
    Calculate price percentiles for each product from a dataframe
    """
    product_tiers = {}

    # Get unique products
    products = df['famille_equipement_produit'].dropna().unique()

    print(f"Analyzing {len(products)} products...")

    for product in products:
        # Get all prices for this product
        prices = df[df['famille_equipement_produit'] == product]['mt_apres_remise_ht_devis'].dropna()

        if len(prices) >= min_samples:
            p30 = prices.quantile(0.3)
            p70 = prices.quantile(0.7)
            p90 = prices.quantile(0.9)

            product_tiers[product] = {
                'p30': round(p30, 2),
                'p70': round(p70, 2),
                'p90': round(p90, 2),
                'count': len(prices)
            }
            print(f"  ✅ {product}: {len(prices)} samples")
        else:
            print(f"  ⚠️ {product}: only {len(prices)} samples (need {min_samples}) - skipping")

    return product_tiers
