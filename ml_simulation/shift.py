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
        Crée un diagramme unique et lisible montrant la distribution binned
        des probabilités base vs transformées.

        Paramètres:
        - comparison_df : DataFrame avec 'customer_id', 'base_prediction', 'new_prediction'
        - bins : nombre de bins (défaut 10)
        - title : titre personnalisé du graphique
        """
        baseline = comparison_df['base_prediction'].values
        transformed = comparison_df['new_prediction'].values

        n_customers = len(comparison_df)
        avg_base = np.mean(baseline)
        avg_new = np.mean(transformed)
        avg_lift = avg_new - avg_base
        extra_conversions_approx = np.sum(transformed) - np.sum(baseline)

        # Création des bins
        # bin_edges = np.linspace(0, 1, bins + 1)
        # bin_labels = [f"{int(e * 100)}-{int(bin_edges[i + 1] * 100)}%" for i, e in enumerate(bin_edges[:-1])]

        # In your plotting function, replace the bin_edges line:
        all_probs = np.concatenate([baseline, transformed])
        bin_edges = np.quantile(all_probs, np.linspace(0, 1, bins + 1))
        bin_labels = [f"{bin_edges[i]:.3f} – {bin_edges[i + 1]:.3f}" for i in range(bins)]

        # Histogrammes
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
        transformed_hist, _ = np.histogram(transformed, bins=bin_edges)

        # Figure
        fig = go.Figure()

        # Barres baseline
        fig.add_trace(
            go.Bar(
                name='Situation actuelle',
                x=bin_labels,
                y=baseline_hist,
                marker_color='#6baed6',
                opacity=0.75,
                text=baseline_hist,
                textposition='auto',
                hovertemplate='%{x}<br>Clients: %{y}<extra></extra>'
            )
        )

        # Barres transformées
        fig.add_trace(
            go.Bar(
                name='Après transformation',
                x=bin_labels,
                y=transformed_hist,
                marker_color='#ff7f0e',
                opacity=0.75,
                text=transformed_hist,
                textposition='auto',
                hovertemplate='%{x}<br>Clients: %{y}<extra></extra>'
            )
        )

        # Lignes moyennes
        fig.add_vline(
            x=avg_base * 100,
            line_dash='dash',
            line_color='#6baed6',
            line_width=2.5,
            annotation_text=f"Moyenne actuelle : {avg_base:.1%}",
            annotation_position="top right",
            annotation_font_size=13,
            annotation_font_color='#6baed6'
        )

        fig.add_vline(
            x=avg_new * 100,
            line_dash='dash',
            line_color='#ff7f0e',
            line_width=2.5,
            annotation_text=f"Moyenne après : {avg_new:.1%}",
            annotation_position="bottom right",
            annotation_font_size=13,
            annotation_font_color='#ff7f0e'
        )

        # Mise en page
        fig.update_layout(
            title_text=(
                f"{title}<br>"
                f"<sup>Gain moyen : {avg_lift:+.1%} | ≈ {extra_conversions_approx:.0f} conversions supplémentaires "
                f"sur {n_customers} clients</sup>"
            ),
            title_x=0.5,
            title_font_size=18,
            xaxis_title="Probabilité de conversion (%)",
            yaxis_title="Nombre de clients",
            bargap=0.15,
            height=550,
            width=900,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            font=dict(size=12)
        )

        fig.update_xaxes(tickangle=-45)
        fig.show()

        # Optionnel : sauvegarde
        # fig.write_image("conversion_shift.png", scale=2)
        # fig.write_html("conversion_shift.html")

        return fig  # Or save: plt.savefig('conversion_shift_bar.png')

    def run(self):
        """
        Implements the pipeline:
        1. Create base features
        2. Get base predictions
        3. Apply transformation to quotes
        4. Create transformed features
        5. Get new predictions
        6. Create comparison DataFrame
        7. Plot bar diagram
        """
        # Step 1: Base features
        with redirect_stdout(io.StringIO()):
            df_base = create_features(self.df_quotes)

        # Step 2: Base predictions
        X_base = df_base.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_base['base_prediction'] = self.model.predict(X_base)

        # Step 3: Transformed quotes
        df_quotes_tr = self.apply_change()

        # Step 4: Transformed features
        with redirect_stdout(io.StringIO()):
            df_tr = create_features(df_quotes_tr)

        # Step 5: New predictions
        X_tr = df_tr.drop(columns=['numero_compte', 'converted'], errors='ignore')
        df_tr['new_prediction'] = self.model.predict(X_tr)

        # Step 6: Comparison DataFrame
        comparison_df = pd.merge(
            df_base[['numero_compte', 'base_prediction']],
            df_tr[['numero_compte', 'new_prediction']],
            on='numero_compte',
            how='inner'
        ).rename(columns={'numero_compte': 'customer_id'})

        # Step 7: Bar diagram
        self.show_diagram(comparison_df)

        return comparison_df