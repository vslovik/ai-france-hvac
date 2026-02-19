import pandas as pd


def sample_diverse_multi_quote(df_candidates, n=5, random_state=42):
    return (
        df_candidates.assign(
            rank_per_product=lambda d: d.groupby('product')['quote_count'].rank(method='first', ascending=False)
        )
        .query('rank_per_product == 1')  # one per product, highest quote_count
        .sample(n=min(n, len(df_candidates)), random_state=random_state)  # if enough diversity
        .pipe(lambda d: pd.concat([
            d,
            df_candidates[~df_candidates['customer_id'].isin(d['customer_id'])]
            .sort_values('quote_count', ascending=False)
            .head(n - len(d))
        ]) if len(d) < n else d)
        .drop(columns=['rank_per_product'])
        .sort_values('quote_count', ascending=False)
        .reset_index(drop=True)
    )


def sample_for_single_quote_customers(df, n=5, random_state=None):
    """
    For single-quote customers:
    - Maximize product diversity (one per product, first occurrence after sorting)
    - If fewer than n unique products → fill with random rows from remaining data
    - No reliance on quote_count → more neutral / representative sampling
    """
    if df.empty:
        return pd.DataFrame()

    # Sort once for stable / reproducible "first" picks (customer_id is fine proxy)
    df_sorted = df[df['product'].notna()].sort_values('customer_id')

    # One row per product → first occurrence
    diverse = df_sorted.drop_duplicates(subset='product', keep='first')

    if len(diverse) >= n:
        # Enough unique products → random sample among them
        return diverse.sample(n=n, random_state=random_state).reset_index(drop=True)

    # Not enough diversity → take all + random fill from the rest
    n_missing = n - len(diverse)

    # Optional: exclude already used customers (but often unnecessary for single-quote)
    # used_customers = diverse['customer_id'].unique()
    # remaining = df_sorted[~df_sorted['customer_id'].isin(used_customers)]

    # Simpler: sample from all remaining rows (since customers are single anyway)
    remaining = df_sorted.drop(diverse.index)  # more efficient than isin

    if remaining.empty:
        return diverse.reset_index(drop=True)

    extra = remaining.sample(
        n=min(n_missing, len(remaining)),
        random_state=random_state
    )

    combined = pd.concat([diverse, extra], ignore_index=True)
    return combined.reset_index(drop=True)


def sample_for_single_quote_customers_(df, n=5, random_state=None):
    """
    For customers who quoted exactly once:
    - Pick the highest quote_count version per product
    - Fall back to next-highest quote_count when not enough products
    - Final sort by quote_count descending → highlights strongest items
    """
    if df.empty:
        return pd.DataFrame()

    ranked = (
        df.assign(
            rank_per_product=lambda d: d.groupby('product')['quote_count']
                                     .rank(method='first', ascending=False)
        )
        .query('rank_per_product == 1')
    )

    if len(ranked) >= n:
        return (
            ranked.sample(n=n, random_state=random_state)
                  .drop(columns=['rank_per_product'])
                  .sort_values('quote_count', ascending=False)
                  .reset_index(drop=True)
        )

    # Not enough unique products → take all best-per-product + top remaining
    n_missing = n - len(ranked)
    used_customers = ranked['customer_id'].unique()   # usually almost all

    remaining = df[~df['customer_id'].isin(used_customers)]
    if remaining.empty:
        remaining = df  # very rare edge-case: fall back to whole set

    extra = (
        remaining.sort_values('quote_count', ascending=False)
                  .head(n_missing)
    )

    return (
        pd.concat([ranked, extra], ignore_index=True)
          .drop(columns=['rank_per_product'], errors='ignore')
          .sort_values('quote_count', ascending=False)
          .reset_index(drop=True)
    )