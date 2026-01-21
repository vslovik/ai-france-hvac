import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc


def get_interactive_roc_curve(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    CREATE FULLY CUSTOMIZED PLOT
    """

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Model ROC curve with custom styling
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',  # 'lines', 'lines+markers', 'markers'
        name=f'Model {model_name} (AUC = {roc_auc:.3f})',
        line=dict(
            color='#FF6B6B',  # Hex color
            width=4,  # Line thickness
            dash='solid'  # 'solid', 'dash', 'dot', 'dashdot'
        ),
        hovertemplate=(
                'FPR: %{x:.3f}<br>' +
                'TPR: %{y:.3f}<br>' +
                '<extra></extra>'  # Hides trace name from hover
        ),
        showlegend=True
    ), row=1, col=1)

    # 2. Random classifier with custom styling
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.500)',
        line=dict(
            color='#4ECDC4',
            width=3,
            dash='dash'
        ),
        showlegend=True
    ), row=1, col=1)

    # 3. Optional: Perfect classifier
    fig.add_trace(go.Scatter(
        x=[0, 0, 1],
        y=[0, 1, 1],
        mode='lines',
        name='Perfect (AUC = 1.000)',
        line=dict(
            color='#45B7D1',
            width=2,
            dash='dot'
        ),
        showlegend=True
    ), row=1, col=1)

    # 4. Customize layout
    fig.update_layout(
        # Size
        width=700,  # pixels
        height=700,

        # Title
        title=dict(
            text="<b>ROC Curve Analysis</b>",
            font=dict(
                family="Arial",
                size=24,
                color="#2C3E50"
            ),
            x=0.5,  # Center title
            xanchor="center"
        ),

        # Axes
        xaxis=dict(
            title=dict(
                text="<b>False Positive Rate</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[-0.02, 1.02],  # Extend slightly beyond 0-1
            gridcolor='lightgray',
            gridwidth=1,
            zerolinecolor='gray',
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),

        yaxis=dict(
            title=dict(
                text="<b>True Positive Rate</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[-0.02, 1.02],
            gridcolor='lightgray',
            gridwidth=1,
            zerolinecolor='gray',
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),

        # Legend
        legend=dict(
            x=0.02,  # Left position
            y=0.98,  # Top position
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14)
        ),

        # Background
        plot_bgcolor='white',
        paper_bgcolor='white',

        # Margins
        margin=dict(l=0, r=0, t=0, b=8),

        # Hover behavior
        hovermode='x unified',  # 'x', 'y', 'closest', False

        # Shapes (add reference areas)
        shapes=[
            # Good performance area
            dict(
                type="rect",
                xref="x", yref="y",
                x0=0, x1=0.2,
                y0=0.8, y1=1,
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            ),
            # Random area
            dict(
                type="rect",
                xref="x", yref="y",
                x0=0, x1=1,
                y0=0, y1=1,
                fillcolor="rgba(128, 128, 128, 0.05)",
                line=dict(width=0),
                layer="below"
            )
        ],

        # Annotations (add text)
        annotations=[
            dict(
                x=0.6,
                y=0.3,
                text=f"<b>AUC = {roc_auc:.3f}</b>",
                showarrow=True,
                arrowhead=2,
                ax=50,
                ay=-40,
                font=dict(size=16, color="#FF6B6B"),
                bgcolor="white",
                bordercolor="#FF6B6B",
                borderwidth=2,
                borderpad=4
            )
        ]
    )

    # 5. Show/hide specific controls
    fig.update_layout(
        modebar=dict(
            # Remove buttons you don't want
            remove=[
                'lasso2d',
                'select2d',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'toggleSpikelines'
            ],
            # Keep only these buttons
            add=[
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ]
        )
    )

    # 7. Optional: Add interactive buttons (e.g., toggle log scale, hide perfect curve)
    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             type="buttons",
    #             direction="right",
    #             x=0.95,
    #             y=1.15,
    #             showactive=True,
    #             buttons=list([
    #                 dict(
    #                     label="Log Scale",
    #                     method="update",
    #                     args=[{"visible": [True, True, True]},
    #                           {"yaxis": {"type": "log"}}]
    #                 ),
    #                 dict(
    #                     label="Linear Scale",
    #                     method="update",
    #                     args=[{"visible": [True, True, True]},
    #                           {"yaxis": {"type": "linear"}}]
    #                 ),
    #                 dict(
    #                     label="Hide Perfect",
    #                     method="update",
    #                     args=[{"visible": [True, True, False]},
    #                           {"title": "ROC Curve (Perfect Hidden)"}]
    #                 )
    #             ]),
    #         )
    #     ]
    # )

    return fig
