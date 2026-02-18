import pandas as pd


def sample_with_product_diversity(df_candidates, n=5, random_state=42):
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