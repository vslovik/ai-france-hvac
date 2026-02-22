import pandas as pd
import numpy as np
import plotly.graph_objects as go
from contextlib import redirect_stdout
import io
from ml_features.features import create_features


class ConversionShiftSimulator:
    def __init__(self, df_quotes, model, show_plot):
        self.df_quotes = df_quotes
        self.model = model
        self.show_plot = show_plot

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

    def get_comparison_df(self):
        # Step 1: Base features
        with redirect_stdout(io.StringIO()):
            df_base = create_features(self.df_quotes)

        # Step 2: Base predictions
        X_base = df_base.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_base['base_prediction'] = self.model.predict_proba(X_base)[:, 1]

        # Step 3: Transformed quotes
        df_quotes_tr = self.apply_change()

        # Step 4: Features on transformed quotes
        with redirect_stdout(io.StringIO()):
            df_tr = create_features(df_quotes_tr)

        # Step 5: New predictions
        X_tr = df_tr.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_tr['new_prediction'] = self.model.predict_proba(X_tr)[:, 1]

        # Step 6: Comparison DataFrame
        comparison_df = pd.merge(
            df_base[['numero_compte', 'base_prediction']],
            df_tr[['numero_compte', 'new_prediction']],
            on='numero_compte', how='inner'
        ).rename(columns={'numero_compte': 'customer_id'})

        return comparison_df

    def run(self):
        """Runs the simulation pipeline"""
        comparison_df = self.get_comparison_df()
        if self.show_plot:
            self.show_diagram(comparison_df)
        return comparison_df