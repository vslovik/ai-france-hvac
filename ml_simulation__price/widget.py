import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class PriceMatchWidget:
    OPTIONS = {
        "Prix actuel": {
            "fam": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Prix actuel (sans réduction)"
        },
        "-10%": {
            "fam": -0.10,
            "emoji": "🔰",
            "color": "#fdae61",
            "title": "Réduction -10%"
        },
        "-15%": {
            "fam": -0.15,
            "emoji": "💰",
            "color": "#2ca02c",
            "title": "Réduction -15%"
        },
        "-20%": {
            "fam": -0.20,
            "emoji": "🏷️",
            "color": "#d62728",
            "title": "Réduction -20%"
        },
    }

    def __init__(self, compute_func, selected_ids):
        self.compute_func = compute_func
        self.selected_ids = selected_ids

    def show(self):
        # ─── Figure factory ───
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
                    go.Bar(x=[f"{product[:8]}"], y=[base_val], marker_color=c_base,
                           text=f"{base_val:.3f}", textposition='auto',
                           hovertemplate=f"Prix actuel<br>€{price:.0f}<br>{base_val:.3f}<extra></extra>"),
                    row=1, col=i + 1
                )

                # Right bar - reduced price
                new_price = price * (1 + info['fam']) if info['fam'] else price
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

        # ─── Widgets ───
        dropdown = widgets.Dropdown(
            options=list(self.OPTIONS.keys()),
            value="Prix actuel",
            description='Réduction :',
            layout={'width': '380px'}
        )

        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                key = dropdown.value
                fam = self.OPTIONS[key]['fam']
                data = self.compute_func(family=fam)
                fig = make_fig(data, key)
                display(fig)

        dropdown.observe(update, names='value')

        # Show UI
        ui = widgets.VBox([
            widgets.HBox([dropdown]),
            output
        ])

        update()  # initial plot
        display(ui)

        # table = wandb.Table(columns=["Scenario", "Plot"])
        #
        # for scen in self.OPTIONS:
        #     fam = self.OPTIONS[scen]["fam"]
        #     data = self.compute_func(family=fam)
        #     fig = make_fig(data, scen)
        #     fig.show()
        #     html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        #
        #     table.add_data(scen, wandb.Html(html))
        #
        # wandb.log({"follow_up_comparison": table})


def show_price_match_widget(compute_func, selected_ids):
    widget = PriceMatchWidget(compute_func, selected_ids)
    widget.show()