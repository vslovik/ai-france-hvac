import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ml_simulation__discount.shift import simulate_discount_conversion_shift


class Graph:
    DISCOUNT_LEVELS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]

    def __init__(self, df_sim, model, selected_ids):
        self.df_sim = df_sim
        self.model = model
        self.selected_ids = selected_ids

    def get_data(self):
        # Calculate base price per customer for discount amounts
        customer_base_price = self.df_sim.groupby('numero_compte')['mt_apres_remise_ht_devis'].sum().to_dict()

        # Filter once
        df_sim_filtered = self.df_sim[self.df_sim['numero_compte'].isin(self.selected_ids)].copy()

        # Run simulations for each discount level
        results = []
        for discount in self.DISCOUNT_LEVELS:
            comparison_df = simulate_discount_conversion_shift(df_sim_filtered, self.model, discount, False)
            comparison_df['discount_pct'] = discount

            # Add discount amount column
            comparison_df['discount_amount'] = comparison_df['customer_id'].map(
                lambda cust_id: customer_base_price.get(cust_id, 0) * (discount / 100)
            )

            results.append(comparison_df)

        # Combine results
        final_results = pd.concat(results, ignore_index=True)
        print(final_results.head())

        # Combine results
        final_results = pd.concat(results, ignore_index=True)
        print(final_results.head())
        return final_results

    def show(self):
        data = self.get_data()

        # Create figure
        fig = go.Figure()

        # Add one line per customer
        for cust_id in self.selected_ids:
            cust_data = data[data['customer_id'] == cust_id].sort_values('discount_pct')

            # Skip if no data for this customer
            if cust_data.empty:
                print(f"Warning: No data for customer {cust_id}")
                continue

            fig.add_trace(go.Scatter(
                x=cust_data['discount_pct'],
                y=cust_data['new_prediction'],
                mode='lines+markers',
                name=str(cust_id),
                hovertemplate=(
                        f"Customer: {cust_id}<br>" +
                        "Discount: %{x}%<br>" +
                        "Probability: %{y:.3f}<br>" +
                        "Δ from baseline: %{customdata:.3f}<br>" +
                        "Discount amount: €%{text:.2f}"
                ),
                customdata=cust_data['new_prediction'] - cust_data['base_prediction'],
                text=cust_data['discount_amount']
            ))

        # Add baseline reference (0% discount)
        for cust_id in self.selected_ids:
            baseline_data = data[
                (data['customer_id'] == cust_id) &
                (data['discount_pct'] == 0)
                ]

            if not baseline_data.empty:
                baseline = baseline_data['new_prediction'].iloc[0]

                fig.add_hline(
                    y=baseline,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"{cust_id} baseline",
                    annotation_position="right",
                    annotation_font_size=10
                )

        # Layout
        fig.update_layout(
            title="Comparative Discount Sensitivity Analysis",
            xaxis_title="Discount Percentage (%)",
            yaxis_title="Conversion Probability",
            hovermode="closest",
            template="plotly_white",
            height=600,
            showlegend=True
        )

        # Add current scenario marker (1.5%)
        fig.add_vline(x=1.5, line_dash="dash", line_color="red",
                      annotation_text="Current: 1.5%", annotation_position="top")

        fig.show()

        print("✓ Comparative discount response curves created!")
        print("✓ Hover over any point to see customer-specific details")
        print("✓ Each line shows a different customer's sensitivity to discounts")


def show_discount_comparison_graph(df_sim, model, selected_ids):
    graph = Graph(df_sim, model, selected_ids)
    graph.show()