import pandas as pd
import numpy as np
from ml_simulation.shift import ConversionShiftSimulator


class PriceConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model, price_change_percentage):
        super().__init__(df_quotes, model)
        self.price_change_percentage = price_change_percentage

    def apply_change(self) -> pd.DataFrame:
        """
        Increases or decreases the net price (mt_apres_remise_ht_devis)
        of the MOST RECENT quote per customer by the given percentage.

        ONLY applied to customers with NO existing discount history.

        Parameters:
        -----------
        df_quotes : pd.DataFrame
            DataFrame with one row per quote (id_devis unique)
        price_change_percentage : float
            Percentage change (e.g. 0.10 = +10%, -0.15 = -15%)

        Behavior:
        - Only the most recent quote per numero_compte is modified
        - Only for customers with NO existing discounts
        - Recency: highest dt_creation_devis → on tie: highest id_devis
        - Other quotes remain unchanged
        - Works with empty DataFrame (returns copy)
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = ['numero_compte', 'dt_creation_devis', 'id_devis', 'mt_apres_remise_ht_devis']
        missing = [c for c in required if c not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Defensive: check uniqueness if this is important for your use case
        if self.df_quotes['id_devis'].duplicated().any():
            raise ValueError("id_devis is not unique — function assumes one row = one quote")

        df = self.df_quotes.copy()

        # Parse dates safely
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Sort: most recent quotes come first
        df_sorted = df.sort_values(
            by=['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)

        # Identify the row to modify (rank 0 = most recent per customer)
        df_sorted['rank'] = df_sorted.groupby('numero_compte').cumcount()
        mask_most_recent = df_sorted['rank'] == 0

        # NEW: Identify customers with NO existing discount history
        if 'mt_remise_exceptionnelle_ht' in df_sorted.columns:
            # Calculate total existing discount per customer
            customer_discount_total = df_sorted.groupby('numero_compte')['mt_remise_exceptionnelle_ht'].transform('sum')
            # Customers with zero or null existing discount
            mask_no_discount = (customer_discount_total.fillna(0) == 0)
        else:
            # If column doesn't exist, assume no discounts exist
            mask_no_discount = True

        # DEBUG: Show filtering
        print(f"\n=== PRICE CHANGE APPLICATION FILTERING ===")
        print(f"Price change: {self.price_change_percentage * 100:+.1f}%")
        print(f"Total customers: {df_sorted['numero_compte'].nunique()}")
        print(f"Customers with no existing discount: {df_sorted[mask_no_discount]['numero_compte'].nunique()}")
        print(f"Most recent quotes (all customers): {mask_most_recent.sum()}")

        # Combine conditions: most recent quote AND customer has no existing discount
        mask_target = mask_most_recent & mask_no_discount

        print(f"Targeted quotes (most recent + no discount history): {mask_target.sum()}")

        if mask_target.sum() == 0:
            print("⚠️ No customers meet the criteria for price changes!")
            # Return unmodified data
            return df_sorted.drop(columns=['rank'], errors='ignore').set_index(df.index).sort_index()

        # Calculate new price only for target rows
        current_price = df_sorted['mt_apres_remise_ht_devis']
        new_price = current_price * (1 + self.price_change_percentage)

        # Calculate price change amount
        price_change_amount = new_price - current_price
        price_change_amount = price_change_amount.where(mask_target, 0.0)

        print(f"\nPrice change stats for targeted customers:")
        print(f"  Number of quotes with price change: {(price_change_amount != 0).sum()}")
        print(f"  Total price change: €{price_change_amount.sum():+.2f}")
        print(f"  Average change per targeted quote: €{price_change_amount[mask_target].mean():+.2f}")

        # IMPORTANT: If price changes and there's a discount column, we need to adjust
        # the discount amount to maintain the same discount PERCENTAGE
        if 'mt_remise_exceptionnelle_ht' in df_sorted.columns and mask_target.any():
            # For targeted quotes, calculate what the discount should be
            # to maintain the same discount percentage relative to original price

            # Get original price before any changes
            original_price = current_price.copy()

            # Get current discount (negative = discount given)
            current_discount = df_sorted['mt_remise_exceptionnelle_ht'].fillna(0.0)

            # Calculate original discount percentage (if price > 0)
            # Note: discount % = abs(discount) / original_price
            original_discount_pct = np.where(
                original_price > 0,
                -current_discount / original_price,  # Negative to positive
                0
            )

            # For targeted quotes, update discount to maintain same percentage
            # New discount = - (original_discount_pct * new_price)
            new_discount = np.where(
                mask_target,
                -original_discount_pct * new_price,
                current_discount
            )

            # Apply the updated discount
            df_sorted['mt_remise_exceptionnelle_ht'] = new_discount

            # Show discount adjustment
            discount_change = new_discount - current_discount
            print(f"\nDiscount adjustment for targeted quotes:")
            print(f"  Original discount sum: €{current_discount[mask_target].sum():.2f}")
            print(f"  New discount sum: €{new_discount[mask_target].sum():.2f}")
            print(f"  Discount change: €{discount_change[mask_target].sum():+.2f}")

        # Apply price change to selected rows
        df_sorted['mt_apres_remise_ht_devis'] = current_price.where(
            ~mask_target,  # keep original when not target
            new_price  # apply new value when target
        )

        # Show before/after totals
        before_sum = current_price.sum()
        after_sum = df_sorted['mt_apres_remise_ht_devis'].sum()
        print(f"\nPrice totals:")
        print(f"  Total before: €{before_sum:.2f}")
        print(f"  Total after: €{after_sum:.2f}")
        print(f"  Net change: €{after_sum - before_sum:+.2f}")

        # Optional: prevent negative prices (uncomment if desired)
        # df_sorted['mt_apres_remise_ht_devis'] = df_sorted['mt_apres_remise_ht_devis'].clip(lower=0)

        # Clean up helper column
        df_sorted = df_sorted.drop(columns=['rank'], errors='ignore')

        # Restore original row order
        return df_sorted.set_index(df.index).sort_index()


def simulate_price_conversion_shift(df_quotes, model, price_change_percentage):
    simulator = PriceConversionShiftSimulator(df_quotes, model, price_change_percentage)
    return simulator.run()


def simulate_price_conversion_shift_all(df_quotes, model, price_changes=(-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10)):
    """
    Simulate multiple price change scenarios and find the best per customer.
    """
    print("\n=== SIMULATING MULTIPLE PRICE SCENARIOS ===")

    results = {}

    for pct_change in price_changes:
        print(f"\n--- Running price change: {pct_change * 100:+.1f}% ---")
        df_result = simulate_price_conversion_shift(df_quotes, model, pct_change)
        results[pct_change] = df_result

    # Merge results and find best per customer
    df_final = results[0][['customer_id', 'base_prediction']].copy()

    for pct, df_res in results.items():
        col_name = f'new_pred_{pct * 100:+.0f}%'
        df_final = df_final.merge(
            df_res[['customer_id', 'new_prediction']].rename(columns={'new_prediction': col_name}),
            on='customer_id',
            how='left'
        )

    # Find best prediction
    new_pred_cols = [c for c in df_final.columns if c.startswith('new_pred_')]
    df_final['best_new_prediction'] = df_final[new_pred_cols].max(axis=1)
    df_final['best_price_change'] = df_final[new_pred_cols].idxmax(axis=1).str.extract(r'([+-]\d+)').astype(float) / 100

    # Summary
    print("\n=== PRICE SCENARIO SUMMARY ===")
    print(f"Average base prediction: {df_final['base_prediction'].mean():.4f}")
    print(f"Average best prediction: {df_final['best_new_prediction'].mean():.4f}")
    print(f"Average lift: {df_final['best_new_prediction'].mean() - df_final['base_prediction'].mean():+.4f}")

    return df_final