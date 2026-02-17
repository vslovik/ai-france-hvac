import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class SalesRepWidget:
    OPTIONS = {
        "Commercial actuel": {
            "rep": None,
            "emoji": "📊",
            "color": "#6baed6",
            "title": "Commercial actuel"
        },
        "MARINA GUYOT": {
            "rep": "marina",
            "emoji": "🟠",
            "color": "#ff7f0e",
            "title": "MARINA GUYOT (Discount-focused)"
        },
        "ELISABETH MACHADO": {
            "rep": "elisabeth",
            "emoji": "🟢",
            "color": "#2ca02c",
            "title": "ELISABETH MACHADO (Value-focused)"
        },
        "Clément TOUZAN": {
            "rep": "clement",
            "emoji": "🔵",
            "color": "#1f77b4",
            "title": "Clément TOUZAN (Neutral)"
        },
    }

    def __init__(self, compute_func, selected_ids):
        self.compute_func = compute_func
        self.selected_ids = selected_ids

    def show(self):

        # ─── Figure factory ───
        def make_fig(data, key):
            info = self.OPTIONS[key]
            is_current = key == "Commercial actuel"

            fig = make_subplots(
                1, len(self.selected_ids),
                subplot_titles=[f"{str(cid)[:8]}<br><sub>{seg[:3]}</sub>"
                                for cid, seg in zip(self.selected_ids, data['segments'])],
                horizontal_spacing=0.14,
                shared_yaxes=True
            )

            for i in range(len(self.selected_ids)):
                base_val = data['base'][i]
                new_val = data['new'][i]
                segment = data['segments'][i]
                current_rep = data['current_reps'][i]
                delta = new_val - base_val

                # Determine colors based on segment
                if segment == 'discount_sensitive':
                    light_color = '#fdae61'
                    dark_color = '#ff7f0e'
                elif segment == 'value_sensitive':
                    light_color = '#98df8a'
                    dark_color = '#2ca02c'
                else:  # neutral
                    light_color = '#6baed6'
                    dark_color = '#1f77b4'

                # Left bar - current rep
                fig.add_trace(
                    go.Bar(x=['Actuel'], y=[base_val],
                           marker_color=light_color,
                           text=f"{base_val:.3f}", textposition='auto',
                           hovertemplate=f"Commercial actuel: {current_rep}<br>{base_val:.3f}<extra></extra>"),
                    row=1, col=i + 1
                )

                # Right bar - with new rep
                bar_color = light_color if is_current or delta == 0 else (dark_color if delta > 0 else '#d62728')

                fig.add_trace(
                    go.Bar(x=[key], y=[new_val],
                           marker_color=bar_color,
                           text=f"{new_val:.3f}" + ("" if is_current else f"<br>{delta:+.3f}"),
                           textposition='auto',
                           hovertemplate=f"{key}<br>{new_val:.3f}" + (
                               "" if is_current else f" ({delta:+.3f})") + "<extra></extra>"),
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
            value="Commercial actuel",
            description='Commercial :',
            layout={'width': '380px'}
        )

        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                key = dropdown.value
                rep = self.OPTIONS[key]['rep']
                data = self.compute_func(rep_type=rep)
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


def show_sales_rep_widget(compute_func, selected_ids):
    widget = SalesRepWidget(compute_func, selected_ids)
    widget.show()