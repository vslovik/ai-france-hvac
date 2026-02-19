import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class DiscountWidget:
    OPTIONS = {
        "Prix actuel": {
            "discount": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Sans remise"
        },
        "1.0%": {
            "discount": 1.0,
            "emoji": "🔰",
            "color": "#fdae61",
            "title": "Remise 1.0%"
        },
        "1.5%": {
            "discount": 1.5,
            "emoji": "💰",
            "color": "#2ca02c",
            "title": "Remise 1.5%"
        },
        "2.0%": {
            "discount": 2.0,
            "emoji": "🏷️",
            "color": "#d62728",
            "title": "Remise 2.0%"
        },
        "2.5%": {
            "discount": 2.5,
            "emoji": "🔥",
            "color": "#ff7f0e",
            "title": "Remise 2.5%"
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
                current_discount = data['current_discounts'][i]
                delta = new_val - base_val

                # Left bar - current price/discount
                fig.add_trace(
                    go.Bar(x=[f"{product[:8]}"], y=[base_val], marker_color=c_base,
                           text=f"{base_val:.3f}", textposition='auto',
                           hovertemplate=f"Prix actuel<br>€{price:.0f}<br>Remise: €{current_discount:.0f}<br>{base_val:.3f}<extra></extra>"),
                    row=1, col=i + 1
                )

                # Right bar - with discount
                if is_current:
                    new_discount = current_discount
                    new_price = price
                else:
                    new_discount = data['discounts'][i]
                    new_price = price * (1 - info['discount'] / 100)

                hover_text = f"{key}<br>€{new_price:.0f}<br>Remise: €{new_discount:.0f}<br>{new_val:.3f}" + (
                    "" if is_current else f" ({delta:+.3f})")

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
            description='Remise :',
            layout={'width': '380px'}
        )

        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                key = dropdown.value
                discount = self.OPTIONS[key]['discount']
                data = self.compute_func(family=discount)
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


def show_discount_widget(compute_func, selected_ids):
    widget = DiscountWidget(compute_func, selected_ids)
    widget.show()