import pandas as pd
import numpy as np
import plotly.graph_objects as go
from contextlib import redirect_stdout
import io
from ml_features.features import create_features


class ConversionShiftSimulator:
    def __init__(self, df_quotes, model):
        self.df_quotes = df_quotes
        self.model = model

    def apply_change(self):
        return self.df_quotes.copy()  # default: no change

    @staticmethod
    def show_diagram(
            comparison_df: pd.DataFrame,
            bins: int = 10,
            title: str = "Shift des probabilités de conversion après transformation"
    ):
        """
        Crée un diagramme montrant la distribution des probabilités de conversion.
        """
        baseline = comparison_df['base_prediction'].values
        transformed = comparison_df['new_prediction'].values

        n_customers = len(comparison_df)
        avg_base = np.mean(baseline)
        avg_new = np.mean(transformed)
        avg_lift = avg_new - avg_base
        extra_conversions_approx = np.sum(transformed) - np.sum(baseline)

        # FIX: Use fixed-width bins from 0 to 1
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_labels = [f"{bin_edges[i] * 100:.0f}-{bin_edges[i + 1] * 100:.0f}%" for i in range(bins)]

        # Histograms with fixed-width bins
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
        transformed_hist, _ = np.histogram(transformed, bins=bin_edges)

        # Calculate percentages for better interpretation
        baseline_pct = (baseline_hist / n_customers * 100).round(1)
        transformed_pct = (transformed_hist / n_customers * 100).round(1)

        # Figure
        fig = go.Figure()

        # Add baseline bars
        fig.add_trace(
            go.Bar(
                name='Situation actuelle',
                x=bin_labels,
                y=baseline_hist,
                marker_color='#6baed6',
                opacity=0.75,
                text=[f"{h} clients<br>({p}%)" for h, p in zip(baseline_hist, baseline_pct)],
                textposition='inside',
                textfont=dict(color='white', size=10),
                hovertemplate='<b>%{x}</b><br>' +
                              'Clients: %{y}<br>' +
                              'Pourcentage: %{text}<extra></extra>'
            )
        )

        # Add transformed bars
        fig.add_trace(
            go.Bar(
                name='Après transformation',
                x=bin_labels,
                y=transformed_hist,
                marker_color='#ff7f0e',
                opacity=0.75,
                text=[f"{h} clients<br>({p}%)" for h, p in zip(transformed_hist, transformed_pct)],
                textposition='inside',
                textfont=dict(color='white', size=10),
                hovertemplate='<b>%{x}</b><br>' +
                              'Clients: %{y}<br>' +
                              'Pourcentage: %{text}<extra></extra>'
            )
        )

        # Add vertical lines for means (convert to percentage for x-axis positioning)
        fig.add_vline(
            x=avg_base * 100,  # Convert to percentage for x-axis
            line_dash='dash',
            line_color='#6baed6',
            line_width=2.5,
            annotation_text=f"Moyenne actuelle : {avg_base:.1%}",
            annotation_position="top",
            annotation_font_size=12,
            annotation_font_color='#6baed6'
        )

        fig.add_vline(
            x=avg_new * 100,
            line_dash='dash',
            line_color='#ff7f0e',
            line_width=2.5,
            annotation_text=f"Moyenne après : {avg_new:.1%}",
            annotation_position="bottom",
            annotation_font_size=12,
            annotation_font_color='#ff7f0e'
        )

        # Highlight the shift
        if avg_lift > 0:
            # Add an arrow or annotation showing the shift direction
            fig.add_annotation(
                x=avg_new * 100,
                y=max(max(baseline_hist), max(transformed_hist)) * 0.9,
                xref="x",
                yref="y",
                text=f"↑ +{avg_lift:.1%}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#2ca02c",
                ax=avg_base * 100,
                ay=max(max(baseline_hist), max(transformed_hist)) * 0.9,
                axref="x",
                ayref="y"
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=(
                    f"{title}<br>"
                    f"<sup>Gain moyen : {avg_lift:+.1%} | ≈ {extra_conversions_approx:.0f} conversions supplémentaires "
                    f"sur {n_customers} clients</sup>"
                ),
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Probabilité de conversion",
                tickangle=-45,
                gridcolor='lightgrey',
                showgrid=True
            ),
            yaxis=dict(
                title="Nombre de clients",
                gridcolor='lightgrey',
                showgrid=True
            ),
            bargap=0.15,
            bargroupgap=0.05,
            height=600,
            width=1000,
            template='plotly_white',
            hovermode='x',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgrey',
                borderwidth=1
            ),
            font=dict(size=12)
        )

        # Add a subtle background color for the area where most customers are
        fig.add_vrect(
            x0=0,
            x1=100,
            fillcolor="lightgrey",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

        fig.show()
        return fig

    def run_(self):
        """ Implements the pipeline: 1. Create base features 2. Get base predictions 3. Apply transformation to quotes 4. Create transformed features 5. Get new predictions 6. Create comparison DataFrame 7. Plot bar diagram """
        # Step 1: Base features
        with redirect_stdout(io.StringIO()):
            df_base = create_features(self.df_quotes)
            # Step 2: Base predictions
        X_base = df_base.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_base['base_prediction'] = self.model.predict_proba(X_base)[:, 1]  # FIX: use predict_proba
        # Step 3: Transformed quotes
        df_quotes_tr = self.apply_change()
        # Step 4: Transformed features
        with redirect_stdout(io.StringIO()):
            df_tr = create_features(df_quotes_tr)
            # Step 5: New predictions
        X_tr = df_tr.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_tr['new_prediction'] = self.model.predict_proba(X_tr)[:, 1]  # FIX: use predict_proba
        # Step 6: Comparison DataFrame
        comparison_df = pd.merge(
            df_base[['numero_compte', 'base_prediction']],
            df_tr[['numero_compte', 'new_prediction']],
            on='numero_compte', how='inner'
        ).rename(columns={'numero_compte': 'customer_id'})
        # Step 7: Bar diagram
        self.show_diagram(comparison_df)
        return comparison_df

    def run(self):
        """ Implements the pipeline with debugging """
        print("\n=== DEBUG: ConversionShiftSimulator.run() ===")

        # Step 1: Base features
        print("\n1. Creating base features...")
        with redirect_stdout(io.StringIO()):
            df_base = create_features(self.df_quotes)

        print(f"   Base features shape: {df_base.shape}")
        # print(f"   Base features columns: {df_base.columns.tolist()}")
        print(f"   Number of customers in base: {df_base['numero_compte'].nunique()}")

        # Step 2: Base predictions
        print("\n2. Getting base predictions...")
        X_base = df_base.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_base['base_prediction'] = self.model.predict_proba(X_base)[:, 1]

        print(f"   Base predictions stats:")
        print(f"     Min: {df_base['base_prediction'].min():.4f}")
        print(f"     Max: {df_base['base_prediction'].max():.4f}")
        print(f"     Mean: {df_base['base_prediction'].mean():.4f}")

        # Step 3: Transformed quotes
        print("\n3. Applying transformation to quotes...")
        df_quotes_tr = self.apply_change()

        print(f"\n4. Creating features on transformed quotes...")
        with redirect_stdout(io.StringIO()):
            df_tr = create_features(df_quotes_tr)

        print(f"   Transformed features shape: {df_tr.shape}")
        print(f"   Number of customers in transformed: {df_tr['numero_compte'].nunique()}")

        # Step 5: Compare features
        print("\n5. Comparing base vs transformed features for same customers...")

        # Merge to compare
        comparison_features = df_base.merge(
            df_tr,
            on='numero_compte',
            suffixes=('_base', '_tr'),
            how='inner'
        )

        print(f"   Customers in both: {len(comparison_features)}")

        # Find numeric columns to compare
        numeric_cols_base = [c for c in df_base.columns if c not in ['numero_compte', 'converted']
                             and pd.api.types.is_numeric_dtype(df_base[c])]

        print("\n   Feature changes:")
        for col in numeric_cols_base[:10]:  # Check first 10 features
            base_col = col + '_base'
            tr_col = col + '_tr'
            if base_col in comparison_features.columns and tr_col in comparison_features.columns:
                if not comparison_features[base_col].equals(comparison_features[tr_col]):
                    diff = (comparison_features[tr_col] - comparison_features[base_col]).abs().sum()
                    if diff > 0:
                        print(f"     {col}: changed (total abs diff: {diff:.4f})")

        # Step 6: New predictions
        print("\n6. Getting new predictions...")
        X_tr = df_tr.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_tr['new_prediction'] = self.model.predict_proba(X_tr)[:, 1]

        print(f"   New predictions stats:")
        print(f"     Min: {df_tr['new_prediction'].min():.4f}")
        print(f"     Max: {df_tr['new_prediction'].max():.4f}")
        print(f"     Mean: {df_tr['new_prediction'].mean():.4f}")

        # Step 7: Comparison DataFrame
        print("\n7. Creating comparison DataFrame...")
        comparison_df = pd.merge(
            df_base[['numero_compte', 'base_prediction']],
            df_tr[['numero_compte', 'new_prediction']],
            on='numero_compte', how='inner'
        ).rename(columns={'numero_compte': 'customer_id'})

        print(f"   Comparison shape: {comparison_df.shape}")
        print(f"   Prediction changes:")
        print(f"     Mean base: {comparison_df['base_prediction'].mean():.4f}")
        print(f"     Mean new: {comparison_df['new_prediction'].mean():.4f}")
        print(f"     Mean lift: {comparison_df['new_prediction'].mean() - comparison_df['base_prediction'].mean():.4f}")

        # Check if any predictions decreased
        decreased = comparison_df['new_prediction'] < comparison_df['base_prediction']
        print(
            f"     Customers with decreased prediction: {decreased.sum()} ({decreased.sum() / len(comparison_df) * 100:.1f}%)")

        print("\n=== END DEBUG ===\n")

        # Step 8: Bar diagram
        self.show_diagram(comparison_df)
        return comparison_df