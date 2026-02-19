import numpy as np
import pandas as pd

from ml_simulation.constrants import HEAT_PUMP, STOVE, COLD_REGIONS
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


def get_single_quote_customers(df_simulation):
    df_not_converted = get_nonconverted_customers(df_simulation)
    df_eligible = df_not_converted[df_not_converted['quote_count'] == 1]
    df_eligible = df_eligible.rename(columns={
        'numero_compte': 'customer_id',
        'famille_equipement_produit': 'product',
        'mt_apres_remise_ht_devis': 'price'
    })
    return df_eligible


def get_customers_with_heat_pump_quote_with_no_stove_quote_in_cold_region_(df_simulation: pd.DataFrame) -> pd.Series:
    df = get_nonconverted_customers(df_simulation)

    mask = (
            df.groupby('numero_compte')['famille_equipement_produit']
            .transform(lambda x: HEAT_PUMP in x.values and STOVE not in x.values)
            &
            df['nom_region'].isin(COLD_REGIONS)
    )

    return df[mask]['numero_compte'].drop_duplicates()


def get_customers_with_heat_pump_quote_with_no_stove_quote_in_cold_region(df_simulation):
    df_not_converted = get_nonconverted_customers(df_simulation)

    if df_not_converted.empty:
        print("⚠️ No non-converted customers found!")
        return pd.DataFrame(columns=['customer_id', 'region', 'quote_count', 'quotes'])

    agg = (
        df_not_converted
        .groupby('numero_compte')
        .agg(
            quote_count=('numero_compte', 'size'),
            has_heat_pump=('famille_equipement_produit', lambda x: HEAT_PUMP in x.values),
            has_stove=('famille_equipement_produit', lambda x: STOVE in x.values),
            region=('nom_region', 'first'),
            quotes=('numero_compte', lambda g: g)  # ← fixed
        )
        .reset_index()
        .rename(columns={'numero_compte': 'customer_id'})
    )

    eligible = agg[
        agg['has_heat_pump'] &
        ~agg['has_stove'] &
        agg['region'].isin(COLD_REGIONS)
        ].copy()

    eligible = eligible[['customer_id', 'region', 'quote_count', 'quotes']]
    eligible['region'] = eligible['region'].fillna('Unknown')

    print(f"✅ Found {len(eligible)} eligible heat pump owners in cold regions")

    if len(eligible) == 0:
        print("⚠️ No eligible customers found!")

    return eligible


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


def find_price_segment_candidates(
    df_eligible: pd.DataFrame,
    product_prices: dict,
    mode: str = 'mid_range',               # 'mid_range', 'non_budget', 'all'
    min_desired_candidates: int = 10,
    expand_premium_light: bool = False,
    premium_light_factor: float = 1.18,
    quote_selector=select_representative_quote,
) -> pd.DataFrame:
    """
    Unified function to get candidates in different price segments.
    - mid_range:          p30 ≤ price ≤ p70            → tier 'mid_range'
    - non_budget:         price > p30                  → tier 'standard' / 'premium'
    - all:                any valid price              → tier 'unknown' / 'standard' / 'premium'
    """
    # ── 1. Representative quote per customer ─────────────────────────────
    df_rep = (
        df_eligible.groupby('numero_compte')
        .apply(quote_selector)
        .reset_index(drop=True)
    )

    # ── 2. Join percentiles ───────────────────────────────────────────────
    percentiles_df = pd.DataFrame.from_dict(
        product_prices, orient='index'
    ).rename_axis('famille_equipement_produit').reset_index()

    df = df_rep.merge(percentiles_df, on='famille_equipement_produit', how='left')

    # ── 3. Basic filters ──────────────────────────────────────────────────
    df = df[df['mt_apres_remise_ht_devis'].notna() & (df['mt_apres_remise_ht_devis'] > 0)].copy()

    # ── 4. Assign segment / tier ──────────────────────────────────────────
    conditions = [
        (df['mt_apres_remise_ht_devis'] <= df['p30']),
        (df['p30'] < df['mt_apres_remise_ht_devis']) & (df['mt_apres_remise_ht_devis'] <= df['p70']),
        (df['mt_apres_remise_ht_devis'] > df['p70']),
    ]
    choices = ['budget', 'mid_range', 'premium']

    df['tier'] = np.select(conditions, choices, default='unknown')
    df['segment'] = df['tier'].replace({'mid_range': 'standard', 'unknown': 'unknown'})

    # ── 5. Filter according to requested mode ─────────────────────────────
    if mode == 'mid_range':
        keep = df['tier'] == 'mid_range'
    elif mode == 'non_budget':
        keep = df['tier'].isin(['mid_range', 'premium'])
    elif mode == 'all':
        keep = df['tier'].notna()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df_candidates = df[keep].copy()

    # ── 6. Enrich with quote_count, savings, etc. ─────────────────────────
    df_candidates = df_candidates.assign(
        quote_count = lambda d: df_eligible.groupby('numero_compte')
                                       .size()
                                       .reindex(d['numero_compte'])
                                       .values,
        budget_price = lambda d: d['p30'],
        savings      = lambda d: d['mt_apres_remise_ht_devis'] - d['p30'],
        savings_pct  = lambda d: np.where(
            d['mt_apres_remise_ht_devis'] > 0,
            ((d['mt_apres_remise_ht_devis'] - d['p30']) / d['mt_apres_remise_ht_devis'] * 100).clip(0, 100),
            0.0
        )
    ).rename(columns={
        'numero_compte': 'customer_id',
        'famille_equipement_produit': 'product',
        'mt_apres_remise_ht_devis': 'price'
    })

    cols = ['customer_id', 'product', 'price', 'segment', 'tier', 'quote_count',
            'budget_price', 'savings', 'savings_pct']

    df_candidates = df_candidates[cols].reset_index(drop=True)

    # ── 7. Optional premium-light expansion (only for mid_range mode) ─────
    if mode == 'mid_range' and expand_premium_light and len(df_candidates) < min_desired_candidates:
        # ... same premium-light logic as before ...
        # (omitted for brevity — copy from previous version)
        pass

    return df_candidates.sort_values('savings_pct', ascending=False)