import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets

from ml_simulation.widget import Widget


class PriceMatchWidget(Widget):
    OPTIONS = {
        "Prix actuel": {
            "scenario": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Prix actuel (sans réduction)"
        },
        "-10%": {
            "scenario": -0.10,
            "emoji": "🔰",
            "color": "#fdae61",
            "title": "Réduction -10%"
        },
        "-15%": {
            "scenario": -0.15,
            "emoji": "💰",
            "color": "#2ca02c",
            "title": "Réduction -15%"
        },
        "-20%": {
            "scenario": -0.20,
            "emoji": "🏷️",
            "color": "#d62728",
            "title": "Réduction -20%"
        },
    }

    def __init__(self, compute_func, selected_ids, log_to_wandb=False):
        super().__init__(compute_func, selected_ids, log_to_wandb)

    def get_make_fig(self):
        def make_fig(data, key):
            info = self.OPTIONS[key]
            is_current = key == "Prix actuel"

            fig = make_subplots(
                1, len(self.selected_ids),
                subplot_titles=[f"{str(cid)[:8]}<br><sub>{p[:10]}</sub>" for cid, p in
                                zip(self.selected_ids, data['products'])],
                horizontal_spacing=0.14,
                shared_yaxes=True
            )

            c_base = '#6baed6'
            c_new = info['color']
            c_down = '#d62728'

            for i in range(len(self.selected_ids)):
                base_val = data['base'][i]
                new_val = data['new'][i]
                product = data['products'][i]
                price = data['prices'][i]
                delta = new_val - base_val

                # Left bar - current price
                fig.add_trace(
                    go.Bar(x=[f"{product[:10]}"], y=[base_val], marker_color=c_base,
                           text=f"{base_val:.3f}", textposition='auto',
                           hovertemplate=f"Prix actuel<br>€{price:.0f}<br>{base_val:.3f}<extra></extra>"),
                    row=1, col=i + 1
                )

                # Right bar - reduced price
                new_price = price * (1 + info['scenario']) if info['scenario'] else price
                hover_text = f"{key}<br>€{new_price:.0f}<br>{new_val:.3f}" + ("" if is_current else f" ({delta:+.3f})")

                fig.add_trace(
                    go.Bar(x=[key], y=[new_val],
                           marker_color=c_new if is_current or delta >= 0 else c_down,
                           text=f"{new_val:.3f}" + ("" if is_current else f"<br>{delta:+.3f}"),
                           textposition='auto',
                           hovertemplate=hover_text + "<extra></extra>"),
                    row=1, col=i + 1
                )

            delta_txt = f"(Δ moyen {data['delta_avg']:+.3f})" if not is_current else ""
            title = f"{info['emoji']} {info['title']} {delta_txt}"

            fig.update_layout(
                title_text=title,
                height=540,
                template="plotly_white",
                barmode='group',
                margin=dict(t=110, b=60, l=50, r=30),
                showlegend=False
            )
            fig.update_yaxes(title_text="Probabilité de conversion", range=[0, 0.9])
            return fig
        return make_fig

    def get_dropdown(self):
        return widgets.Dropdown(
            options=list(self.OPTIONS.keys()),
            value="Prix actuel",
            description='Réduction :',
            layout={'width': '380px'}
        )


def show_price_match_widget(compute_func, selected_ids, log_to_wandb=False):
    widget = PriceMatchWidget(compute_func, selected_ids, log_to_wandb)
    widget.show()