import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class BudgetAlternativeWidget:
    OPTIONS = {
        "Prix actuel": {
            "scenario": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Prix actuel (premium/standard)"
        },
        "Version Budget": {
            "scenario": "budget",
            "emoji": "💰",
            "color": "#2ca02c",
            "title": "Version budget (-30% à -50%)"
        }
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
                subplot_titles=[f"{str(cid)[:10]}<br><sub>{p[:10]}</sub>" for cid, p in
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
                tier = data['tiers'][i]
                delta = new_val - base_val

                # Budget price if available
                budget_price = data['budget_prices'][i] if 'budget_prices' in data else price
                price_display = f"€{price:.0f}" if is_current else f"€{price:.0f} → €{budget_price:.0f}"

                # Left bar - current price
                fig.add_trace(
                    go.Bar(x=[f"{product[:8]}"], y=[base_val], marker_color=c_base,
                           text=f"{base_val:.3f}", textposition='auto',
                           hovertemplate=f"Prix actuel<br>€{price:.0f}<br>{tier}<br>{base_val:.3f}<extra></extra>"),
                    row=1, col=i + 1
                )

                # Right bar - budget version
                hover_text = f"{key}<br>{price_display}<br>{new_val:.3f}" + ("" if is_current else f" ({delta:+.3f})")

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
            description='Scénario :',
            layout={'width': '380px'}
        )

        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                key = dropdown.value
                scenario = self.OPTIONS[key]['scenario']
                data = self.compute_func(family=scenario)
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


def show_budget_alternative_widget(compute_func, selected_ids):
    widget = BudgetAlternativeWidget(compute_func, selected_ids)
    widget.show()