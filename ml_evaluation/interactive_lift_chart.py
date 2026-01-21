import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_interactive_lift_chart(y_test, y_pred_proba, model_name: str = "", n_deciles: int = 10) -> go.Figure:
    """
    Create interactive Lift Chart with custom styling.
    """
    # Create deciles
    df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred_proba
    })

    df['decile'] = pd.qcut(df['pred'], n_deciles, labels=False, duplicates='drop')

    # Calculate metrics per decile
    decile_stats = df.groupby('decile').agg({
        'true': ['count', 'sum', 'mean'],
        'pred': 'mean'
    })
    decile_stats.columns = ['count', 'positives', 'response_rate', 'avg_score']

    # Calculate lift
    overall_response = df['true'].mean()
    decile_stats['response_rate'] = decile_stats['positives'] / decile_stats['count']
    decile_stats['lift'] = decile_stats['response_rate'] / overall_response

    # Sort by decile (highest scores first)
    decile_stats = decile_stats.sort_index(ascending=False)
    decile_stats['cumulative_positives'] = decile_stats['positives'].cumsum()
    decile_stats['cumulative_lift'] = decile_stats['cumulative_positives'] / \
                                      (decile_stats['count'].cumsum() * overall_response)

    # Create figure with secondary y-axis
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Lift bars (primary axis)
    fig.add_trace(go.Bar(
        x=list(range(1, len(decile_stats) + 1)),
        y=decile_stats['lift'],
        name='Lift per Decile',
        marker_color='#FF6B6B',
        opacity=0.8,
        hovertemplate=(
                'Decile: %{x}<br>' +
                'Lift: %{y:.2f}x<br>' +
                'Response Rate: %{customdata:.1%}<br>' +
                '<extra></extra>'
        ),
        customdata=decile_stats['response_rate'],
        showlegend=True
    ))

    # 2. Cumulative Lift line (secondary axis)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(decile_stats) + 1)),
        y=decile_stats['cumulative_lift'],
        mode='lines+markers',
        name='Cumulative Lift',
        line=dict(color='#4ECDC4', width=4),
        marker=dict(size=8, symbol='circle'),
        yaxis='y2',
        hovertemplate=(
                'Top %{x} deciles<br>' +
                'Cumulative Lift: %{y:.2f}x<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 3. Baseline (lift = 1)
    fig.add_trace(go.Scatter(
        x=[0.5, len(decile_stats) + 0.5],
        y=[1, 1],
        mode='lines',
        name='Baseline (Lift = 1)',
        line=dict(color='#45B7D1', width=2, dash='dash'),
        showlegend=True
    ))

    # Update layout with secondary y-axis
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text=f"<b>Lift Chart (Top Decile Lift = {decile_stats['lift'].iloc[0]:.2f}x)</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Decile (1 = Highest Scores)</b>",
                font=dict(size=16, color="#34495E")
            ),
            tickmode='linear',
            tick0=1,
            dtick=1,
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="<b>Lift</b>",
                font=dict(size=16, color="#34495E")
            ),
            gridcolor='lightgray',
            gridwidth=1,
            zerolinecolor='gray',
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis2=dict(
            title=dict(
                text="<b>Cumulative Lift</b>",
                font=dict(size=16, color="#4ECDC4")
            ),
            overlaying='y',
            side='right',
            showgrid=False
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
        margin=dict(l=0, r=60, t=0, b=8),
        hovermode='x unified',
        barmode='group'
    )

    return fig


def get_interactive_lift_chart_two_color(y_test, y_pred_proba, model_name: str = "", n_deciles: int = 10) -> go.Figure:
    """
    Lift chart with TWO COLORS:
    - Bars ABOVE baseline (lift > 1) in GREEN
    - Bars BELOW baseline (lift < 1) in RED
    """

    # Create deciles
    df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred_proba
    })

    df['decile'] = pd.qcut(df['pred'], n_deciles, labels=False, duplicates='drop')

    # Calculate metrics per decile (sorted highest scores first)
    decile_stats = df.groupby('decile').agg({
        'true': ['count', 'sum', 'mean'],
        'pred': 'mean'
    })
    decile_stats.columns = ['count', 'positives', 'response_rate', 'avg_score']
    decile_stats = decile_stats.sort_index(ascending=False)

    # Calculate lift
    overall_response = df['true'].mean()
    decile_stats['lift'] = decile_stats['response_rate'] / overall_response

    # Create separate arrays for above/below baseline
    deciles = list(range(1, len(decile_stats) + 1))
    lifts = decile_stats['lift'].values

    # Split into above and below baseline
    above_mask = lifts >= 1
    below_mask = lifts < 1

    above_deciles = np.array(deciles)[above_mask]
    above_lifts = lifts[above_mask]

    below_deciles = np.array(deciles)[below_mask]
    below_lifts = lifts[below_mask]

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Bars ABOVE baseline (lift >= 1) - GREEN
    if len(above_deciles) > 0:
        fig.add_trace(go.Bar(
            x=above_deciles,
            y=above_lifts - 1,  # Height above baseline
            base=1,  # Start from baseline
            name='Lift > 1 (Good)',
            marker_color='#27AE60',  # Green
            marker_line_color='#1E8449',
            marker_line_width=1,
            opacity=0.8,
            hovertemplate=(
                    'Decile %{x}<br>' +
                    'Lift: %{y:.2f}x<br>' +
                    'Response Rate: %{customdata:.1%}<br>' +
                    '<extra>Above Baseline</extra>'
            ),
            customdata=decile_stats.loc[above_mask, 'response_rate'],
            showlegend=True
        ))

    # 2. Bars BELOW baseline (lift < 1) - RED
    if len(below_deciles) > 0:
        fig.add_trace(go.Bar(
            x=below_deciles,
            y=below_lifts - 1,  # Height above baseline (negative)
            base=1,  # Start from baseline
            name='Lift < 1 (Poor)',
            marker_color='#E74C3C',  # Red
            marker_line_color='#B03A2E',
            marker_line_width=1,
            opacity=0.8,
            hovertemplate=(
                    'Decile %{x}<br>' +
                    'Lift: %{y:.2f}x<br>' +
                    'Response Rate: %{customdata:.1%}<br>' +
                    '<extra>Below Baseline</extra>'
            ),
            customdata=decile_stats.loc[below_mask, 'response_rate'],
            showlegend=True
        ))

    # 3. Baseline line (lift = 1)
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text="Baseline (Lift = 1)",
        annotation_position="bottom right"
    )

    # 4. Add target line (e.g., lift = 2 for top decile)
    target_lift = 2.0
    fig.add_hline(
        y=target_lift,
        line_dash="dot",
        line_color="#F39C12",
        line_width=1,
        annotation_text=f"Target (Lift = {target_lift})",
        annotation_position="top right",
        annotation_font_color="#F39C12"
    )

    # 5. Add data labels on bars
    for decile in deciles:
        lift = decile_stats.loc[decile_stats.index[decile - 1], 'lift']
        color = '#27AE60' if lift >= 1 else '#E74C3C'

        fig.add_annotation(
            x=decile,
            y=lift + 0.05 if lift >= 1 else lift - 0.1,
            text=f"{lift:.2f}x",
            showarrow=False,
            font=dict(
                size=11,
                color='white' if lift >= 1 else 'white',
                weight='bold'
            ),
            bgcolor=color,
            bordercolor='white',
            borderwidth=1,
            borderpad=2,
            opacity=0.9
        )

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text=f"<b>Lift Chart: {model_name}</b><br>"
                 f"<span style='font-size:14px; color:#7F8C8D'>Top Decile Lift = {decile_stats['lift'].iloc[0]:.2f}x | Baseline = 1.0x</span>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Decile (1 = Highest Predicted Scores)</b>",
                font=dict(size=16, color="#34495E")
            ),
            tickmode='array',
            tickvals=deciles,
            ticktext=[f"D{i}" for i in deciles],
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="<b>Lift (Response Rate / Overall Average)</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[0, max(decile_stats['lift'].max() * 1.1, 2.5)],
            gridcolor='lightgray',
            gridwidth=1,
            zerolinecolor='gray',
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12),
            tickformat='.1f'
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=2,
            font=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=100, b=8),
        hovermode='x unified',
        barmode='relative'  # Stack bars relative to baseline
    )

    # Add summary statistics
    top_decile_lift = decile_stats['lift'].iloc[0]
    avg_lift_top5 = decile_stats['lift'].head(5).mean()
    total_lift = decile_stats['lift'].mean()

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Performance Summary:</b><br>"
            f"• Top Decile Lift: <span style='color:#27AE60'>{top_decile_lift:.2f}x</span><br>"
            f"• Avg Top 5 Deciles: <span style='color:#27AE60'>{avg_lift_top5:.2f}x</span><br>"
            f"• Overall Lift: {total_lift:.2f}x<br>"
            f"• Baseline: 1.0x"
        ),
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#2C3E50",
        borderwidth=2,
        borderpad=8
    )

    # Add interpretation
    if top_decile_lift >= 3:
        rating = "🏆 EXCELLENT"
        color = "#27AE60"
    elif top_decile_lift >= 2:
        rating = "✅ VERY GOOD"
        color = "#2ECC71"
    elif top_decile_lift >= 1.5:
        rating = "👍 GOOD"
        color = "#F1C40F"
    elif top_decile_lift >= 1:
        rating = "⚠️ FAIR"
        color = "#E67E22"
    else:
        rating = "❌ POOR"
        color = "#E74C3C"

    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"<span style='color:{color}'><b>{rating}</b></span>",
        showarrow=False,
        font=dict(size=16, weight='bold'),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=color,
        borderwidth=2,
        borderpad=10
    )

    return fig


def get_interactive_lift_chart_gradient(y_test, y_pred_proba, model_name: str = "", n_deciles: int = 10) -> go.Figure:
    """
    Lift chart with GRADIENT colors based on lift value.
    """

    # Create deciles
    df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred_proba
    })

    df['decile'] = pd.qcut(df['pred'], n_deciles, labels=False, duplicates='drop')

    # Calculate metrics
    decile_stats = df.groupby('decile').agg({
        'true': ['count', 'sum', 'mean'],
        'pred': 'mean'
    })
    decile_stats.columns = ['count', 'positives', 'response_rate', 'avg_score']
    decile_stats = decile_stats.sort_index(ascending=False)

    # Calculate lift
    overall_response = df['true'].mean()
    decile_stats['lift'] = decile_stats['response_rate'] / overall_response

    # Create gradient colors based on lift value
    max_lift = decile_stats['lift'].max()
    min_lift = decile_stats['lift'].min()

    def get_color(lift_value):
        # Normalize lift to 0-1 range
        if lift_value >= 1:
            # Green gradient for lift >= 1
            intensity = min((lift_value - 1) / (max_lift - 1), 1) if max_lift > 1 else 0
            # From light green to dark green
            r = int(39 + (33 - 39) * intensity)
            g = int(174 + (97 - 174) * intensity)
            b = int(96 + (25 - 96) * intensity)
        else:
            # Red gradient for lift < 1
            intensity = (1 - lift_value) / (1 - min_lift) if min_lift < 1 else 0
            # From light red to dark red
            r = int(231 + (176 - 231) * intensity)
            g = int(76 + (58 - 76) * intensity)
            b = int(60 + (46 - 60) * intensity)
        return f'rgb({r},{g},{b})'

    colors = [get_color(lift) for lift in decile_stats['lift']]

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # Single bar trace with different colors for each bar
    fig.add_trace(go.Bar(
        x=list(range(1, len(decile_stats) + 1)),
        y=decile_stats['lift'].values,
        name='Lift per Decile',
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.85,
        hovertemplate=(
                'Decile %{x}<br>' +
                'Lift: %{y:.2f}x<br>' +
                'Response Rate: %{customdata:.1%}<br>' +
                '<extra></extra>'
        ),
        customdata=decile_stats['response_rate'].values,
        showlegend=False
    ))

    # Add baseline
    fig.add_hline(
        y=1,
        line_dash="solid",
        line_color="black",
        line_width=2,
        annotation_text="Baseline = 1.0",
        annotation_position="bottom right",
        annotation_font_size=12
    )

    # Add color scale legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=15,
            color='#27AE60',
            symbol='square'
        ),
        name='High Lift (> 1.5)',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=15,
            color='#F1C40F',
            symbol='square'
        ),
        name='Medium Lift (1.0-1.5)',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=15,
            color='#E74C3C',
            symbol='square'
        ),
        name='Low Lift (< 1.0)',
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text=f"<b>Lift Chart with Gradient Colors: {model_name}</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Decile (1 = Highest Scores)</b>",
                font=dict(size=16, color="#34495E")
            ),
            tickmode='array',
            tickvals=list(range(1, len(decile_stats) + 1)),
            ticktext=[f"D{i}" for i in range(1, len(decile_stats) + 1)],
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="<b>Lift</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[0, max(decile_stats['lift'].max() * 1.2, 2.0)],
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
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=2,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=80, b=8),
        hovermode='x unified'
    )

    return fig

