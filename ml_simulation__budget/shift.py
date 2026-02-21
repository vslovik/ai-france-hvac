
import pandas as pd
import numpy as np
from ml_simulation.data import Simulation
from ml_simulation.shift import ConversionShiftSimulator


class BudgetConversionShiftSimulator(ConversionShiftSimulator):
    def __init__(self, df_quotes, model):
        super().__init__(df_quotes, model)

        # ONLY touch these products:
        self.budget_winners = [
            'Chaudière',
            'Poêle',
            'Climatisation',
            'ECS : Chauffe-eau ou adoucisseur',
            'Photovoltaïque'
        ]

        # ONLY premium for VMC
        self.premium_products = [
            'Produit VMC'
        ]

        # ALL OTHER PRODUCTS = NO CHANGE!
        # This includes:
        # - Pompe à chaleur (heat pumps) ← NO TOUCH!
        # - Appareil hybride
        # - Autres
        # - Emetteur de chauffage ou chappe
        # - Plomberie Sanitaire
        # - Fumisterie

    def apply_change(self) -> pd.DataFrame:
        """
        ONLY modifies:
        - BUDGET for boilers, stoves, AC, water heaters, solar
        - PREMIUM for VMC

        ALL OTHER PRODUCTS = NO CHANGE!
        Heat pumps are completely untouched.
        """
        if self.df_quotes.empty:
            return self.df_quotes.copy()

        required = [
            'numero_compte',
            'dt_creation_devis',
            'id_devis',
            'mt_apres_remise_ht_devis',
            'famille_equipement_produit'
        ]
        missing = [c for c in required if c not in self.df_quotes.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        df = self.df_quotes.copy()

        # Parse dates safely
        df['dt_creation_devis'] = pd.to_datetime(df['dt_creation_devis'], errors='coerce')

        # Sort: most recent quotes first
        df_sorted = df.sort_values(
            by=['numero_compte', 'dt_creation_devis', 'id_devis'],
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)

        # Identify most recent quote per customer
        df_sorted['rank'] = df_sorted.groupby('numero_compte').cumcount()
        mask_most_recent = df_sorted['rank'] == 0

        # ── ONLY modify specific products ───────────────────────────────
        def get_strategy_price(row):
            """Only modify budget winners and premium products"""
            if not mask_most_recent.loc[row.name]:
                return row['mt_apres_remise_ht_devis']

            price = row['mt_apres_remise_ht_devis']
            product = row['famille_equipement_produit']

            # BUDGET for winners
            if product in self.budget_winners:
                if product in Simulation.PRODUCT_TIERS:
                    target = Simulation.PRODUCT_TIERS[product].get('p30', price * 0.7)
                else:
                    target = price * 0.7

                # Apply max 30% reduction cap
                reduction = 1 - (target / price) if price > 0 else 0
                max_reduction = 0.3
                if reduction > max_reduction:
                    new_price = price * (1 - max_reduction)
                else:
                    new_price = target

                # if price != new_price:
                #     print(f"  ✓ BUDGET for {product}: €{price:.0f} → €{new_price:.0f}")
                return new_price

            # PREMIUM for VMC
            elif product in self.premium_products:
                if product in Simulation.PRODUCT_TIERS:
                    target = Simulation.PRODUCT_TIERS[product].get('p10', price * 1.2)
                else:
                    target = price * 1.2

                # if price != target:
                #     print(f"  ⭐ PREMIUM for {product}: €{price:.0f} → €{target:.0f}")
                return target

            # EVERYTHING ELSE = NO CHANGE!
            else:
                return price

        # Apply strategy-based pricing
        df_sorted['mt_apres_remise_ht_devis'] = df_sorted.apply(get_strategy_price, axis=1)

        # Adjust discounts proportionally for modified quotes
        if 'mt_remise_exceptionnelle_ht' in df_sorted.columns:
            # Find which rows were modified
            original_prices = df['mt_apres_remise_ht_devis'].values
            new_prices = df_sorted['mt_apres_remise_ht_devis'].values
            price_changed = original_prices != new_prices

            if price_changed.any():
                current_discount = df_sorted['mt_remise_exceptionnelle_ht'].fillna(0)

                # Calculate original discount percentage
                discount_pct = np.where(
                    original_prices > 0,
                    -current_discount / original_prices,
                    0
                )

                # New discount to maintain same percentage
                new_discount = -discount_pct * new_prices

                # Apply only where price changed
                df_sorted['mt_remise_exceptionnelle_ht'] = np.where(
                    price_changed,
                    new_discount,
                    current_discount
                )

        # Clean up
        df_sorted = df_sorted.drop(columns=['rank'], errors='ignore')

        # Restore original row order
        return df_sorted.set_index(df.index).sort_index()


def simulate_budget_conversion_shift(df_quotes, model):
    """
    ONLY modifies:
    - Budget for boilers, stoves, AC, water heaters, solar
    - Premium for VMC

    ALL OTHER PRODUCTS = NO CHANGE!
    Heat pumps are completely untouched.
    """
    print("\n" + "=" * 70)
    print("BUDGET SCENARIO - ONLY TOUCHING RIGHT PRODUCTS")
    print("=" * 70)
    print("✅ Budget winners: Chaudière, Poêle, Climatisation, ECS, Photovoltaïque")
    print("✅ Premium: VMC")
    print("❌ ALL OTHERS: NO CHANGE (including Heat Pumps!)")
    print("=" * 70)

    # Show what will be modified
    df_copy = df_quotes.copy()
    total_customers = df_copy['numero_compte'].nunique()

    budget_customers = df_copy[df_copy['famille_equipement_produit'].isin([
        'Chaudière', 'Poêle', 'Climatisation',
        'ECS : Chauffe-eau ou adoucisseur', 'Photovoltaïque'
    ])]['numero_compte'].nunique()

    premium_customers = df_copy[df_copy['famille_equipement_produit'].isin(['Produit VMC'])]['numero_compte'].nunique()

    untouched = total_customers - budget_customers - premium_customers

    print(f"\nCustomers being modified:")
    print(f"  Budget:  {budget_customers} customers")
    print(f"  Premium: {premium_customers} customers")
    print(f"  Total modified: {budget_customers + premium_customers} customers")
    print(f"  UNTOUCHED: {untouched} customers (including all Heat Pumps!)")

    simulator = BudgetConversionShiftSimulator(df_quotes, model)
    return simulator.run()
