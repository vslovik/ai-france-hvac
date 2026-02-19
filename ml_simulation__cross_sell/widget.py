import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

from ml_simulation__cross_sell.data import CrossSellSimulation


class CrossSellWidget:
    OPTIONS = {
        "Situation actuelle": {
            "family": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Situation actuelle – sans cross-sell"
        },
        "Poêle": {
            "family": "Poêle",
            "emoji": "🔥",
            "color": "#2ca02c",
            "title": "Heat Pump → Poêle (recommandé)"
        },
        "Climatisation": {
            "family": "Climatisation",
            "emoji": "❄️",
            "color": "#ff7f0e",
            "title": "Heat Pump → Climatisation"
        },
        "ECS": {
            "family": "ECS : Chauffe-eau ou adoucisseur",
            "emoji": "💧",
            "color": "#1f77b4",
            "title": "Heat Pump → ECS"
        },
    }

    def __init__(self, compute_func, selected_ids):
        self.compute_func = compute_func
        self.selected_ids = selected_ids

    def show(self):
        # ─── Figure factory ───
        def make_figure(data, selected_key):
            info = self.OPTIONS[selected_key]
            is_current = (selected_key == "Situation actuelle")

            fig = make_subplots(
                rows=1, cols=len(self.selected_ids),
                subplot_titles=[f"{str(cid)[:10]}<br><sub>{r}</sub>"
                                for cid, r in zip(self.selected_ids, data['regions'])],
                horizontal_spacing=0.15,
                shared_yaxes=True
            )

            color_base = '#6baed6'
            color_new = info.get("color", "#2ca02c")
            color_down = '#d62728'

            for i in range(len(self.selected_ids)):
                b = data['base'][i]
                n = data['new'][i]
                delta = n - b

                # Left bar – current situation
                fig.add_trace(
                    go.Bar(
                        x=['Actuel'],
                        y=[b],
                        marker_color=color_base,
                        text=f"{b:.3f}",
                        textposition='auto',
                        hovertemplate=f"Actuel<br>{b:.3f}<extra></extra>",
                        name="Situation actuelle"
                    ),
                    row=1, col=i + 1
                )

                # Right bar – with cross-sell
                fig.add_trace(
                    go.Bar(
                        x=[selected_key],
                        y=[n],
                        marker_color=color_new if is_current or delta >= 0 else color_down,
                        text=f"{n:.3f}" + ("" if is_current else f"<br>{delta:+.3f}"),
                        textposition='auto',
                        hovertemplate=(
                                f"{selected_key}<br>{n:.3f}" +
                                ("" if is_current else f" ({delta:+.3f})") +
                                "<extra></extra>"
                        ),
                        name=f"+ {selected_key}" if not is_current else "Situation actuelle"
                    ),
                    row=1, col=i + 1
                )

            delta_text = f"(Δ moyen {data['delta_avg']:+.3f})" if not is_current else ""
            title = f"Heat Pump → {info.get('emoji', '')} {info['title']} {delta_text}"

            fig.update_layout(
                title=dict(text=title, font_size=18),
                height=520,
                template="plotly_white",
                barmode='group',
                margin=dict(t=110, b=60, l=50, r=30),
                showlegend=False
            )

            fig.update_yaxes(
                title_text="Probabilité de conversion",
                range=[0, max(0.85, max(data['new']) * 1.12)]
            )
            fig.update_xaxes(title_text="")

            return fig

        # ─── Widgets ───
        dropdown = widgets.Dropdown(
            options=list(self.OPTIONS.keys()),
            value="Situation actuelle",
            description='Scénario :',
            layout=widgets.Layout(width='420px')
        )

        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                opt_key = dropdown.value
                family = self.OPTIONS[opt_key]["family"]
                data = self.compute_func(family=family)
                fig = make_figure(data, opt_key)
                display(fig)

        dropdown.observe(update, names='value')

        # Initial plot
        update()

        # Layout
        display(widgets.VBox([
            widgets.HBox([dropdown]),
            output
        ]))


def show_cross_sell_widget(compute_func, selected_ids):
    widget = CrossSellWidget(compute_func, selected_ids)
    widget.show()