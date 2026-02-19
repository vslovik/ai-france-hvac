import numpy as np
import pandas as pd
from ml_simulation.util import get_product_price_tiers


def get_nonconverted_customers(df_simulation):
    # 1. Find non-converted customers
    sim_conv = df_simulation.groupby('numero_compte')['fg_devis_accepte'].max()
    non_converted = sim_conv[sim_conv == 0].index
    print(f"Non-converted customers: {len(non_converted)}")

    # 2. Filter the original dataframe once
    df_nonconv = df_simulation[df_simulation['numero_compte'].isin(non_converted)].copy()

    # 3. Add quote count per customer
    df_nonconv['quote_count'] = df_nonconv.groupby('numero_compte')['numero_compte'].transform('count')

    # # 4. Keep customers with at least 1 quote
    df_eligible = df_nonconv[df_nonconv['quote_count'] >= 1].copy()
    return df_eligible


def get_single_quote_customers(df_eligible):
    df_not_converted = get_nonconverted_customers(df_eligible)
    df_eligible = df_not_converted[df_not_converted['quote_count'] == 1]
    df_eligible = df_eligible.rename(columns={
        'numero_compte': 'customer_id',
        'famille_equipement_produit': 'product',
        'mt_apres_remise_ht_devis': 'price'
    })
    return df_eligible


def get_mid_range_quote_customers(df_eligible):
    product_prices = get_product_price_tiers(df_eligible)

    # Pre-filter only customers whose representative price is in range
    df_rep = df_eligible.groupby('numero_compte').first().reset_index()

    df_with_pct = df_rep.merge(
        pd.DataFrame.from_dict(product_prices, orient='index'),
        left_on='famille_equipement_produit',
        right_index=True,
        how='left'
    )

    df_candidates = df_with_pct[
        (df_with_pct['mt_apres_remise_ht_devis'] >= df_with_pct['p30']) &
        (df_with_pct['mt_apres_remise_ht_devis'] <= df_with_pct['p70']) &
        df_with_pct['p30'].notna()
        ].copy()

    if 'quote_count' not in df_eligible.columns:
        df_candidates['quote_count'] = df_eligible.groupby('numero_compte').size().reindex(
            df_candidates['numero_compte']).values

    df_candidates['tier'] = 'mid_range'
    df_candidates['budget_price'] = df_candidates['p30']
    df_candidates['savings'] = df_candidates['mt_apres_remise_ht_devis'] - df_candidates['p30']
    df_candidates['savings_pct'] = (df_candidates['savings'] / df_candidates['mt_apres_remise_ht_devis'] * 100).fillna(
        0)

    df_candidates = df_candidates.rename(columns={
        'numero_compte': 'customer_id',
        'famille_equipement_produit': 'product',
        'mt_apres_remise_ht_devis': 'price'
    })[['customer_id', 'product', 'price', 'tier', 'quote_count',
        'budget_price', 'savings', 'savings_pct']]

    print(f"🎯 MID-RANGE candidates (p30–p70): {len(df_candidates):,}")
    return df_candidates


def select_representative_quote(group: pd.DataFrame) -> pd.Series:
    """
    Choose which quote to use as representative for a customer group.
    Customize this function based on business needs.
    """
    # Option 1: Cheapest quote (most common for price-sensitive targeting)
    return group.loc[group['mt_apres_remise_ht_devis'].idxmin()]

    # Option 2: Most recent quote (uncomment if you have a date column)
    # return group.sort_values('date_devis', ascending=False).iloc[0]

    # Option 3: First in current order (your original behavior)
    # return group.iloc[0]


def find_mid_range_price_candidates(
        df_eligible: pd.DataFrame,
        product_prices: dict,
        min_desired_candidates: int = 10,
        expand_premium_light: bool = False,
        premium_light_factor: float = 1.18,
        quote_selector=select_representative_quote,
) -> pd.DataFrame:
    """
    Identify mid-range (p30–p70) customers, with optional fallback to
    'premium-light' tier (just above p70) when too few candidates are found.

    Parameters:
    - df_eligible: DataFrame with eligible quotes
    - product_prices: dict like {'Product': {'p30': 400, 'p70': 700}, ...}
    - min_desired_candidates: Expand if fewer than this many mid-range found
    - expand_premium_light: Whether to broaden to p70–(p70 × factor)
    - premium_light_factor: Multiplier above p70 (e.g. 1.18 ≈ loose p85 proxy)
    - quote_selector: Function that takes a group and returns one row

    Returns:
    - DataFrame of candidates with tier, savings, etc. — sorted by opportunity

    Usage examples
        # Basic (mid-range only)
        df_mid_only = find_mid_range_price_candidates(df_eligible, product_prices)

        # With expansion
        df_expanded = find_mid_range_price_candidates(
            df_eligible,
            product_prices,
            min_desired_candidates=15,
            expand_premium_light=True,
            premium_light_factor=1.20   # more aggressive expansion
        )

        # Switch to cheapest quote strategy
        df_cheapest = find_mid_range_price_candidates(
            df_eligible,
            product_prices,
            quote_selector=lambda g: g.loc[g['mt_apres_remise_ht_devis'].idxmin()]
        )
    """
    # ── Step 1: Get one representative quote per customer ────────────────
    df_rep = (
        df_eligible.groupby('numero_compte')
        .apply(quote_selector)
        .reset_index(drop=True)
    )

    # ── Step 2: Join percentiles ─────────────────────────────────────────
    percentiles_df = pd.DataFrame.from_dict(
        product_prices, orient='index'
    ).rename_axis('famille_equipement_produit').reset_index()

    df_with_ranges = df_rep.merge(
        percentiles_df,
        on='famille_equipement_produit',
        how='left'
    )

    # ── Step 3: Filter strict mid-range (p30 ≤ price ≤ p70) ──────────────
    mid_mask = (
            (df_with_ranges['mt_apres_remise_ht_devis'] >= df_with_ranges['p30']) &
            (df_with_ranges['mt_apres_remise_ht_devis'] <= df_with_ranges['p70']) &
            df_with_ranges['p30'].notna() &
            (df_with_ranges['mt_apres_remise_ht_devis'] > 0)
    )

    df_mid = df_with_ranges[mid_mask].copy()

    # ── Step 4: Enrich mid-range candidates ──────────────────────────────
    df_mid = df_mid.assign(
        quote_count=lambda d: df_eligible.groupby('numero_compte')
        .size()
        .reindex(d['numero_compte'])
        .values,
        tier='mid_range',
        budget_price=lambda d: d['p30'],
        savings=lambda d: d['mt_apres_remise_ht_devis'] - d['p30'],
        savings_pct=lambda d: np.where(
            d['mt_apres_remise_ht_devis'] > 0,
            (d['savings'] / d['mt_apres_remise_ht_devis']) * 100,
            0.0
        )
    ).rename(columns={
        'numero_compte': 'customer_id',
        'famille_equipement_produit': 'product',
        'mt_apres_remise_ht_devis': 'price'
    })

    # Select final columns
    cols = ['customer_id', 'product', 'price', 'tier', 'quote_count',
            'budget_price', 'savings', 'savings_pct']
    df_candidates = df_mid[cols].copy()

    # ── Step 5: Optional expansion to premium-light tier ─────────────────
    if expand_premium_light and len(df_candidates) < min_desired_candidates:
        print(f"⚠️ Only {len(df_candidates)} mid-range → expanding to premium-light...")

        already_selected = set(df_candidates['customer_id'])

        premium_mask = (
                (df_with_ranges['mt_apres_remise_ht_devis'] > df_with_ranges['p70']) &
                (df_with_ranges['mt_apres_remise_ht_devis'] <= df_with_ranges['p70'] * premium_light_factor) &
                df_with_ranges['p70'].notna() &
                (df_with_ranges['mt_apres_remise_ht_devis'] > 0) &
                (~df_with_ranges['numero_compte'].isin(already_selected))
        )

        df_premium = df_with_ranges[premium_mask].copy()

        if not df_premium.empty:
            df_premium = df_premium.assign(
                quote_count=lambda d: df_eligible.groupby('numero_compte')
                .size()
                .reindex(d['numero_compte'])
                .values,
                tier='premium_light',
                budget_price=lambda d: d['p30'],
                savings=lambda d: d['mt_apres_remise_ht_devis'] - d['p30'],
                savings_pct=lambda d: np.where(
                    d['mt_apres_remise_ht_devis'] > 0,
                    (d['savings'] / d['mt_apres_remise_ht_devis']) * 100,
                    0.0
                )
            ).rename(columns={
                'numero_compte': 'customer_id',
                'famille_equipement_produit': 'product',
                'mt_apres_remise_ht_devis': 'price'
            })

            df_candidates = pd.concat([df_candidates, df_premium[cols]], ignore_index=True)
            print(f" → Added {len(df_premium)} premium-light → total {len(df_candidates)}")
        else:
            print(" → No premium-light candidates found.")

    # ── Final touch: sort by most interesting opportunities first ────────
    if not df_candidates.empty:
        df_candidates = df_candidates.sort_values(
            ['savings_pct', 'quote_count'],
            ascending=[False, False]
        ).reset_index(drop=True)

    return df_candidates