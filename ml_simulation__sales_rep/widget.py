import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class SalesRepWidget:
    # Color mapping based on segment
    SEGMENT_COLORS = {
        'price_sensitive': {
            'light': '#fdae61',  # Light orange
            'dark': '#ff7f0e',  # Dark orange
            'name': '💰',
            'emoji': '🟠'  # Orange circle emoji for dropdown
        },
        'value_sensitive': {
            'light': '#98df8a',  # Light green
            'dark': '#2ca02c',  # Dark green
            'name': '🌟',
            'emoji': '🟢'  # Green circle emoji for dropdown
        },
        'neutral': {
            'light': '#6baed6',  # Light blue
            'dark': '#1f77b4',  # Dark blue
            'name': '🔵',
            'emoji': '🔵'  # Blue circle emoji for dropdown
        }
    }

    REP_OPTIONS = {
        "Commercial actuel": {
            "rep": None,
            "emoji": "📊",
            "color_emoji": "⬜",  # White circle for current
            "display": "⬜ 📊 Commercial actuel",
            "title": "Commercial actuel"
        },
        "MARINA GUYOT": {
            "rep": "marina",
            "emoji": "💰",
            "color_emoji": "🟠",  # Orange circle matches discount-sensitive
            "display": "🟠 💰 MARINA GUYOT (Discount-focused)",
            "title": "MARINA GUYOT (Discount-focused)"
        },
        "ELISABETH MACHADO": {
            "rep": "elisabeth",
            "emoji": "🌟",
            "color_emoji": "🟢",  # Green circle matches value-sensitive
            "display": "🟢 🌟 ELISABETH MACHADO (Value-focused)",
            "title": "ELISABETH MACHADO (Value-focused)"
        },
        "Clément TOUZAN": {
            "rep": "clement",
            "emoji": "🔵",
            "color_emoji": "🔵",  # Blue circle matches neutral
            "display": "🔵 🔹 Clément TOUZAN (Neutral)",
            "title": "Clément TOUZAN (Neutral)"
        },
    }

    def __init__(self, compute_func, selected_ids):
        self.compute_func = compute_func
        self.selected_ids = selected_ids

    def show(self):
        def make_figure(data, selected_key):
            info = self.REP_OPTIONS[selected_key]
            is_current = (selected_key == "Commercial actuel")

            fig = make_subplots(
                rows=1, cols=len(self.selected_ids),
                subplot_titles=[f"{str(cid)[:8]}<br><sub>{data['regions'][i]}</sub>"
                                for i, cid in enumerate(self.selected_ids)],
                horizontal_spacing=0.15,
                shared_yaxes=True
            )

            for i in range(len(self.selected_ids)):
                b = data['base'][i]
                n = data['new'][i]
                segment = data['segments'][i]
                delta = n - b

                # Get colors based on segment
                colors = self.SEGMENT_COLORS[segment]

                # Left bar – current situation (always light color)
                fig.add_trace(
                    go.Bar(
                        x=['Actuel'],
                        y=[b],
                        marker_color=colors['light'],
                        text=f"{b:.3f}",
                        textposition='auto',
                        hovertemplate=(
                            f"<b>{self.selected_ids[i][:8]}</b><br>"
                            f"Segment: {segment}<br>"
                            f"Commercial: {data['current_reps'][i]}<br>"
                            f"Probabilité: {b:.3f}<extra></extra>"
                        )
                    ),
                    row=1, col=i + 1
                )

                # Right bar – with new rep
                # Dark if improved, light if not
                bar_color = colors['dark'] if delta > 0 else colors['light']

                fig.add_trace(
                    go.Bar(
                        x=[selected_key],
                        y=[n],
                        marker_color=bar_color,
                        text=f"{n:.3f}" + ("" if is_current else f"<br>{delta:+.3f}"),
                        textposition='auto',
                        hovertemplate=(
                                f"{selected_key}<br>{n:.3f}" +
                                ("" if is_current else f" ({delta:+.3f})") +
                                "<extra></extra>"
                        )
                    ),
                    row=1, col=i + 1
                )

            delta_text = f"(Δ moyen {data['delta_avg']:+.3f})" if not is_current else ""

            # Use color_emoji in title for visual consistency
            title = f"{info['color_emoji']} {info['emoji']} {info['title']} {delta_text}"

            fig.update_layout(
                title=dict(text=title, font_size=18),
                height=540,
                template="plotly_white",
                barmode='group',
                margin=dict(t=110, b=60, l=50, r=30),
                showlegend=False
            )

            fig.update_yaxes(
                title_text="Probabilité de conversion",
                range=[0, 0.9]
            )
            fig.update_xaxes(title_text="")

            return fig

        # Widgets - Use display field with color emoji
        dropdown = widgets.Dropdown(
            options=[(info['display'], key) for key, info in self.REP_OPTIONS.items()],
            value="Commercial actuel",
            description='Commercial :',
            layout={'width': '500px'}  # Wider for colored emoji
        )

        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                key = dropdown.value
                rep = self.REP_OPTIONS[key]['rep']
                data = self.compute_func(family=rep)
                fig = make_figure(data, key)
                display(fig)

        dropdown.observe(update, names='value')

        # Initial plot
        update()

        # Layout
        display(widgets.VBox([
            widgets.HBox([dropdown]),
            output
        ]))


def show_sales_rep_widget(compute_func, selected_ids):
    widget = SalesRepWidget(compute_func, selected_ids)
    widget.show()