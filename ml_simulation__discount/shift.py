import pandas as pd

from ml_simulation.shift import ConversionShiftSimulator


class DiscountConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, discount_percentage):
        super().__init__(df_quotes, model)
        self.discount_percentage = discount_percentage

    def apply_change(self) -> pd.DataFrame:
        """
        Applies a discount ONLY to the most recent quote per customer (numero_compte)
        AND ONLY for customers with NO existing discount history.

        Recency rule:
          - Highest dt_creation_devis first
          - If dates are equal → highest id_devis wins

        Preconditions (assumed / enforced):
          - Each row = one unique quote (id_devis is unique across the whole dataframe)
          - mt_apres_remise_ht_devis = total net price of that quote (no line items)
          - mt_remise_exceptionnelle_ht = existing discount amount (if column exists)

        The discount is applied directly to that single row:
          - added to mt_remise_exceptionnelle_ht
          - subtracted from mt_apres_remise_ht_devis
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = ['numero_compte', 'dt_creation_devis', 'id_devis', 'mt_apres_remise_ht_devis']
        missing = [c for c in required if c not in self.df_quotes]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Defensive check based on your assertion
        if self.df_quotes['id_devis'].duplicated().any():
            raise ValueError("id_devis is not unique — the function assumes one row = one quote")

        df = self.df_quotes.copy()

        # Prepare sorting keys
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Sort: most recent quotes first (date desc + id desc on ties)
        df_sorted = df.sort_values(
            by=['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)

        # Identify the most recent quote per customer
        df_sorted['rank'] = df_sorted.groupby('numero_compte').cumcount()
        mask_most_recent = df_sorted['rank'] == 0

        # NEW: Identify customers with NO existing discount history
        # Check if mt_remise_exceptionnelle_ht exists and sum it per customer
        if 'mt_remise_exceptionnelle_ht' in df_sorted.columns:
            # Calculate total existing discount per customer
            customer_discount_total = df_sorted.groupby('numero_compte')['mt_remise_exceptionnelle_ht'].transform('sum')
            # Customers with zero or null existing discount
            mask_no_discount = (customer_discount_total.fillna(0) == 0)
        else:
            # If column doesn't exist, assume no discounts exist
            mask_no_discount = True

        # DEBUG: Show filtering
        print(f"\n=== DISCOUNT APPLICATION FILTERING ===")
        print(f"Total customers: {df_sorted['numero_compte'].nunique()}")
        print(f"Customers with no existing discount: {df_sorted[mask_no_discount]['numero_compte'].nunique()}")
        print(f"Most recent quotes (all customers): {mask_most_recent.sum()}")

        # Combine conditions: most recent quote AND customer has no existing discount
        mask_target = mask_most_recent & mask_no_discount

        print(f"Targeted quotes (most recent + no discount history): {mask_target.sum()}")

        if mask_target.sum() == 0:
            print("⚠️ No customers meet the criteria for new discounts!")
            # Return unmodified data
            return df_sorted.drop(columns=['rank'], errors='ignore').set_index(df.index).sort_index()

        # Calculate discount amount — only for target rows
        discount_amount = df_sorted['mt_apres_remise_ht_devis'] * (self.discount_percentage / 100.0)
        discount_amount = discount_amount.where(mask_target, 0.0)

        print(f"\nDiscount stats for targeted customers:")
        print(f"  Number of quotes receiving discount: {(discount_amount > 0).sum()}")
        print(f"  Total discount amount: €{discount_amount.sum():.2f}")
        print(f"  Average discount per targeted quote: €{discount_amount[mask_target].mean():.2f}")

        # Apply changes — only to the selected quotes
        if 'mt_remise_exceptionnelle_ht' in df_sorted.columns:
            current_remise = df_sorted['mt_remise_exceptionnelle_ht'].fillna(0.0)
            df_sorted['mt_remise_exceptionnelle_ht'] = current_remise - discount_amount

            # Show remise changes
            print(f"\nRemise column update:")
            print(f"  Original remise sum: €{current_remise.sum():.2f}")
            print(f"  New remise sum: €{df_sorted['mt_remise_exceptionnelle_ht'].sum():.2f}")

        # Update price after discount
        before_price_sum = df_sorted['mt_apres_remise_ht_devis'].sum()
        df_sorted['mt_apres_remise_ht_devis'] = (
                df_sorted['mt_apres_remise_ht_devis'] - discount_amount
        )
        after_price_sum = df_sorted['mt_apres_remise_ht_devis'].sum()

        print(f"\nPrice update:")
        print(f"  Total price before: €{before_price_sum:.2f}")
        print(f"  Total price after: €{after_price_sum:.2f}")
        print(f"  Total discount applied: €{before_price_sum - after_price_sum:.2f}")
        print(f"  Expected discount: €{discount_amount.sum():.2f}")

        # Optional: prevent negative prices (business rule choice)
        # df_sorted['mt_apres_remise_ht_devis'] = df_sorted['mt_apres_remise_ht_devis'].clip(lower=0)

        # Clean up temporary columns
        df_sorted = df_sorted.drop(columns=['rank'], errors='ignore')

        # Return in original row order
        return df_sorted.set_index(df.index).sort_index()


def simulate_discount_conversion_shift(df_quotes, model, discount_percentage):
    simulator = DiscountConversionShiftSimulator(df_quotes, model, discount_percentage)
    return simulator.run()


def simulate_discount_conversion_shift_all(df_quotes, model, discount_values = (0.0, 0.015, 0.02, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):

    # Dictionary to store results: {discount: comparison_df}
    results = {}

    # Run simulation for each parameter
    for disc in discount_values:
        print(f"Running simulation with discount = {disc * 100:.1f}% ...")
        df_result = simulate_discount_conversion_shift(df_quotes, model, disc)
        results[disc] = df_result

    # ── Merge all results and keep the MAX new_prediction per customer ────────

    # Start with the base (discount=0) as reference
    df_final = results[0.0][['customer_id', 'base_prediction']].copy()

    # Add new_prediction from each scenario
    for disc, df_res in results.items():
        col_name = f'new_pred_{disc * 100:.0f}%'
        df_final = df_final.merge(
            df_res[['customer_id', 'new_prediction']].rename(columns={'new_prediction': col_name}),
            on='customer_id',
            how='left'
        )

    # Now compute the maximum across all new_prediction columns
    new_pred_cols = [c for c in df_final.columns if c.startswith('new_pred_')]
    df_final['best_new_prediction'] = df_final[new_pred_cols].max(axis=1)

    # Optional: also record which discount gave the best result
    df_final['best_discount_pct'] = df_final[new_pred_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float) / 100

    # Clean up (optional - keep only essential columns)
    df_final = df_final[['customer_id', 'base_prediction', 'best_new_prediction', 'best_discount_pct']]

    # Show result
    print("\nFinal DataFrame with best prediction per customer:")
    print(df_final.head(10))

    # Summary statistics
    print("\nAverage base prediction:   {:.4f}".format(df_final['base_prediction'].mean()))
    print("Average best prediction: {:.4f}".format(df_final['best_new_prediction'].mean()))
    print("Average lift:            {:.4f} ({:+.2%})".format(
        df_final['best_new_prediction'].mean() - df_final['base_prediction'].mean(),
        df_final['best_new_prediction'].mean() - df_final['base_prediction'].mean()
    ))
    print("Number of customers with improvement:",
          (df_final['best_new_prediction'] > df_final['base_prediction']).sum())

    return df_final