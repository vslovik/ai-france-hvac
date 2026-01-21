from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, auc
import plotly.graph_objects as go
import numpy as np


def get_interactive_pr_curve(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Create interactive Precision-Recall Curve with custom styling.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Calculate baseline (prevalence of positive class)
    prevalence = np.mean(y_test)

    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Model PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'Model {model_name} (AP = {pr_auc:.3f})',
        line=dict(
            color='#FF6B6B',
            width=4,
            dash='solid'
        ),
        hovertemplate=(
                'Recall: %{x:.3f}<br>' +
                'Precision: %{y:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 2. Baseline (no skill)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[prevalence, prevalence],
        mode='lines',
        name=f'Baseline (AP = {prevalence:.3f})',
        line=dict(
            color='#4ECDC4',
            width=3,
            dash='dash'
        ),
        showlegend=True
    ))

    # 3. Perfect classifier
    fig.add_trace(go.Scatter(
        x=[0, 1, 1],
        y=[1, 1, prevalence],
        mode='lines',
        name='Perfect (AP = 1.000)',
        line=dict(
            color='#45B7D1',
            width=2,
            dash='dot'
        ),
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text="<b>Precision-Recall Curve Analysis</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Recall (Sensitivity)</b>",
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
        yaxis=dict(
            title=dict(
                text="<b>Precision</b>",
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
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=8),
        hovermode='x unified',
        annotations=[
            dict(
                x=0.6,
                y=0.3,
                text=f"<b>AP = {pr_auc:.3f}</b>",
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

    return fig