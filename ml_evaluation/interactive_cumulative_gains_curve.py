import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


def get_interactive_cumulative_gains_curve(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Create interactive Cumulative Gains Curve with custom styling.
    """
    # Sort by predicted probability
    df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred_proba
    }).sort_values('pred', ascending=False).reset_index(drop=True)

    # Calculate cumulative metrics
    total_positives = df['true'].sum()
    df['cumulative_positives'] = df['true'].cumsum()
    df['population_percentile'] = (df.index + 1) / len(df)
    df['gain'] = df['cumulative_positives'] / total_positives

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Model Gains curve
    fig.add_trace(go.Scatter(
        x=df['population_percentile'],
        y=df['gain'],
        mode='lines',
        name=f'Model {model_name}',
        line=dict(
            color='#FF6B6B',
            width=4,
            dash='solid'
        ),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.1)',
        hovertemplate=(
                'Top %{x:.1%} of population<br>' +
                'Captures %{y:.1%} of positives<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 2. Random model (diagonal)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Model',
        line=dict(
            color='#4ECDC4',
            width=3,
            dash='dash'
        ),
        showlegend=True
    ))

    # 3. Perfect model
    perfect_x = [0, df['true'].mean(), 1]
    perfect_y = [0, 1, 1]
    fig.add_trace(go.Scatter(
        x=perfect_x,
        y=perfect_y,
        mode='lines',
        name='Perfect Model',
        line=dict(
            color='#45B7D1',
            width=2,
            dash='dot'
        ),
        showlegend=True
    ))

    # 4. Add decile markers
    deciles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for decile in deciles:
        idx = int(len(df) * decile)
        if idx < len(df):
            decile_gain = df.loc[idx, 'gain'] if idx < len(df) else 1.0
            fig.add_trace(go.Scatter(
                x=[decile],
                y=[decile_gain],
                mode='markers',
                marker=dict(
                    color='#FFA726',
                    size=8,
                    symbol='circle'
                ),
                name=f'Decile {int(decile * 100)}%',
                showlegend=False,
                hovertemplate=(
                        f'Top {decile:.0%}<br>' +
                        f'Captures {decile_gain:.1%} of positives<br>' +
                        '<extra></extra>'
                )
            ))

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text="<b>Cumulative Gains Curve</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Population Percentile (Sorted by Score)</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[0, 1],
            tickformat='.0%',
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="<b>Cumulative Gain (% of Positives Captured)</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[0, 1.02],
            tickformat='.0%',
            gridcolor='lightgray',
            gridwidth=1,
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
        shapes=[
            # Add reference lines for common deciles
            dict(
                type="line",
                xref="x", yref="paper",
                x0=0.2, x1=0.2,
                y0=0, y1=1,
                line=dict(color="gray", width=1, dash="dot")
            ),
            dict(
                type="line",
                xref="x", yref="paper",
                x0=0.5, x1=0.5,
                y0=0, y1=1,
                line=dict(color="gray", width=1, dash="dot")
            )
        ]
    )

    return fig