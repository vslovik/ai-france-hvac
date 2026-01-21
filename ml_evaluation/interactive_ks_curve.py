import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def get_interactive_ks_curve(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Create interactive KS Curve with correct implementation.
    KS = max|FPR(s) - TPR(s)| where s is the threshold
    """
    from sklearn.metrics import roc_curve

    # Get FPR and TPR at various thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Calculate KS statistic (max difference between TPR and FPR)
    ks_values = tpr - fpr
    ks_statistic = ks_values.max()
    ks_threshold_idx = ks_values.argmax()
    ks_threshold = thresholds[ks_threshold_idx]

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. True Positive Rate (Sensitivity) curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=tpr,
        mode='lines',
        name='True Positive Rate (Sensitivity)',
        line=dict(
            color='#FF6B6B',  # Red
            width=4
        ),
        hovertemplate=(
                'Threshold: %{x:.3f}<br>' +
                'TPR: %{y:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 2. False Positive Rate curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=fpr,
        mode='lines',
        name='False Positive Rate',
        line=dict(
            color='#4ECDC4',  # Teal
            width=4
        ),
        hovertemplate=(
                'Threshold: %{x:.3f}<br>' +
                'FPR: %{y:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 3. KS Difference curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=ks_values,
        mode='lines',
        name=f'KS Statistic (max = {ks_statistic:.3f})',
        line=dict(
            color='#FFA726',  # Orange
            width=3,
            dash='dash'
        ),
        hovertemplate=(
                'Threshold: %{x:.3f}<br>' +
                'KS: %{y:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 4. KS Maximum point
    fig.add_trace(go.Scatter(
        x=[ks_threshold],
        y=[ks_statistic],
        mode='markers+text',
        name=f'KS Max',
        marker=dict(
            color='#D32F2F',
            size=15,
            symbol='diamond'
        ),
        text=[f'KS={ks_statistic:.3f}<br>Threshold={ks_threshold:.3f}'],
        textposition='top right',
        textfont=dict(size=12, color='#D32F2F'),
        hovertemplate=(
                'Threshold: %{x:.3f}<br>' +
                'KS: %{y:.3f}<br>' +
                'TPR: ' + f'{tpr[ks_threshold_idx]:.3f}<br>' +
                'FPR: ' + f'{fpr[ks_threshold_idx]:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text=f"<b>Kolmogorov-Smirnov Curve (KS = {ks_statistic:.3f})</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Threshold (Cut-off Probability)</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[thresholds.min() - 0.02, thresholds.max() + 0.02],
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="<b>Rate / KS Statistic</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[-0.02, 1.02],
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
        margin=dict(l=0, r=0, t=60, b=8),
        hovermode='x unified',
        # Add reference lines
        shapes=[
            # Vertical line at KS threshold
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=ks_threshold,
                x1=ks_threshold,
                y0=0,
                y1=1,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dot"
                )
            ),
            # Horizontal line at KS value
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=ks_statistic,
                y1=ks_statistic,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dot"
                )
            )
        ],
        # Add annotation for interpretation
        annotations=[
            dict(
                x=0.98,
                y=0.02,
                xref="paper",
                yref="paper",
                text=f"<b>Interpretation:</b><br>KS = {ks_statistic:.3f}",
                showarrow=False,
                font=dict(size=12, color="#2C3E50"),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4
            ),
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text="<b>KS measures maximum separation between positive and negative class distributions</b>",
                showarrow=False,
                font=dict(size=11, color="#7F8C8D"),
                align="center"
            )
        ]
    )

    # Add table-like interpretation guide
    ks_interpretation = {
        0.0: "No discrimination",
        0.2: "Poor discrimination",
        0.4: "Fair discrimination",
        0.6: "Good discrimination",
        0.8: "Very good discrimination",
        1.0: "Perfect discrimination"
    }

    # Find the closest interpretation
    closest_ks = min(ks_interpretation.keys(), key=lambda x: abs(x - ks_statistic))

    # Add interpretation annotation
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"<b>KS = {ks_statistic:.3f}</b><br>{ks_interpretation[closest_ks]}",
        showarrow=False,
        font=dict(size=12, color="#2C3E50"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#FF6B6B",
        borderwidth=2,
        borderpad=8
    )

    return fig


def get_interactive_ks_cumulative_plot(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Traditional KS plot showing cumulative distributions of scores for positive and negative classes.
    CORRECTED VERSION with proper axis orientation.
    """
    import numpy as np

    # Separate scores by class
    pos_scores = y_pred_proba[y_test == 1]
    neg_scores = y_pred_proba[y_test == 0]

    # Sort all unique scores for CDF calculation
    all_scores = np.sort(np.unique(np.concatenate([pos_scores, neg_scores])))

    # Calculate empirical CDFs
    def ecdf(x, data):
        return np.searchsorted(np.sort(data), x, side='right') / len(data)

    pos_cdf = ecdf(all_scores, pos_scores)  # P(score ≤ x | positive)
    neg_cdf = ecdf(all_scores, neg_scores)  # P(score ≤ x | negative)

    # Calculate KS statistic (max vertical distance between CDFs)
    ks_statistic = np.max(np.abs(pos_cdf - neg_cdf))
    ks_idx = np.argmax(np.abs(pos_cdf - neg_cdf))
    ks_threshold = all_scores[ks_idx]

    # For KS plot, we want 1 - CDF = P(score ≥ x)
    pos_survival = 1 - pos_cdf  # P(score ≥ x | positive)
    neg_survival = 1 - neg_cdf  # P(score ≥ x | negative)

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Positive class survival function (1 - CDF)
    fig.add_trace(go.Scatter(
        x=all_scores,
        y=pos_survival,
        mode='lines',
        name='Positive Class',
        line=dict(
            color='#FF6B6B',  # Red
            width=4
        ),
        hovertemplate=(
                'Score: %{x:.3f}<br>' +
                '% Positive ≥ score: %{y:.1%}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 2. Negative class survival function (1 - CDF)
    fig.add_trace(go.Scatter(
        x=all_scores,
        y=neg_survival,
        mode='lines',
        name='Negative Class',
        line=dict(
            color='#4ECDC4',  # Teal
            width=4
        ),
        hovertemplate=(
                'Score: %{x:.3f}<br>' +
                '% Negative ≥ score: %{y:.1%}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 3. KS Difference (vertical distance)
    ks_diff = pos_survival - neg_survival  # Or could be neg_survival - pos_survival

    fig.add_trace(go.Scatter(
        x=all_scores,
        y=ks_diff,
        mode='lines',
        name=f'KS Difference (max = {ks_statistic:.3f})',
        line=dict(
            color='#FFA726',  # Orange
            width=3,
            dash='dash'
        ),
        hovertemplate=(
                'Score: %{x:.3f}<br>' +
                'Difference: %{y:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 4. KS Maximum point
    fig.add_trace(go.Scatter(
        x=[ks_threshold],
        y=[ks_statistic],
        mode='markers+text',
        marker=dict(
            color='#D32F2F',
            size=15,
            symbol='diamond'
        ),
        text=[f'KS={ks_statistic:.3f}<br>Score={ks_threshold:.3f}'],
        textposition='top center',
        textfont=dict(size=12, color='#D32F2F'),
        hovertemplate=(
                f'Score: {ks_threshold:.3f}<br>' +
                f'KS: {ks_statistic:.3f}<br>' +
                f'% Positive ≥ score: {pos_survival[ks_idx]:.1%}<br>' +
                f'% Negative ≥ score: {neg_survival[ks_idx]:.1%}<br>' +
                '<extra></extra>'
        ),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text=f"<b>Kolmogorov-Smirnov Plot (KS = {ks_statistic:.3f})</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Predicted Score / Probability</b>",
                font=dict(size=16, color="#34495E")
            ),
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12),
            range=[all_scores.min() - 0.02, all_scores.max() + 0.02]
        ),
        yaxis=dict(
            title=dict(
                text="<b>Proportion of Class with Score ≥ X</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[-0.02, 1.02],
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
        margin=dict(l=0, r=0, t=60, b=8),
        hovermode='x unified',
        # Add vertical line at KS threshold
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=ks_threshold,
                x1=ks_threshold,
                y0=0,
                y1=1,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dot"
                )
            ),
            # Horizontal line at 0 for KS difference reference
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                line=dict(
                    color="black",
                    width=1,
                    dash="dot"
                )
            )
        ],
        # Add interpretation
        annotations=[
            dict(
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                text=(
                    f"<b>KS = {ks_statistic:.3f}</b><br>"
                    f"at score = {ks_threshold:.3f}<br><br>"
                    f"<b>Interpretation:</b><br>"
                    f"Good model if lines are<br>"
                    f"well separated"
                ),
                showarrow=False,
                font=dict(size=12, color="#2C3E50"),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=8
            )
        ]
    )

    return fig


def get_interactive_ks_comprehensive_plot(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Comprehensive KS visualization with CORRECT orientation.
    Shows 1-CDF (survival function) for better visualization of separation.
    """
    import numpy as np
    from plotly.subplots import make_subplots

    # Separate scores by class
    pos_scores = y_pred_proba[y_test == 1]
    neg_scores = y_pred_proba[y_test == 0]

    # Create grid of scores for smooth plotting
    score_min = min(pos_scores.min(), neg_scores.min())
    score_max = max(pos_scores.max(), neg_scores.max())
    all_scores = np.linspace(score_min, score_max, 500)

    # Calculate empirical survival function (1 - CDF) = P(score ≥ x)
    def survival_function(x, data):
        return np.mean(data >= x)

    pos_survival = np.array([survival_function(s, pos_scores) for s in all_scores])
    neg_survival = np.array([survival_function(s, neg_scores) for s in all_scores])

    # Calculate KS statistic (max vertical distance between survival functions)
    ks_diff = pos_survival - neg_survival
    ks_statistic = np.max(np.abs(ks_diff))
    ks_idx = np.argmax(np.abs(ks_diff))
    ks_threshold = all_scores[ks_idx]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"<b>Score Distributions</b>",
            f"<b>Survival Functions (1 - CDF) - KS = {ks_statistic:.3f}</b>"
        ),
        vertical_spacing=0.15,
        row_heights=[0.45, 0.55]
    )

    # 1. TOP: Score distributions (histograms)

    # Positive class histogram
    fig.add_trace(
        go.Histogram(
            x=pos_scores,
            name='Positive Class',
            marker_color='#FF6B6B',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability density',
            hovertemplate=(
                    'Score: %{x:.3f}<br>' +
                    'Density: %{y:.3f}<br>' +
                    '<extra>Positive Class</extra>'
            )
        ),
        row=1, col=1
    )

    # Negative class histogram
    fig.add_trace(
        go.Histogram(
            x=neg_scores,
            name='Negative Class',
            marker_color='#4ECDC4',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability density',
            hovertemplate=(
                    'Score: %{x:.3f}<br>' +
                    'Density: %{y:.3f}<br>' +
                    '<extra>Negative Class</extra>'
            )
        ),
        row=1, col=1
    )

    # Vertical line at KS threshold
    fig.add_vline(
        x=ks_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"KS threshold = {ks_threshold:.3f}",
        annotation_position="top right",
        row=1, col=1
    )

    # 2. BOTTOM: Survival functions (1 - CDF)

    # Positive class survival function
    fig.add_trace(
        go.Scatter(
            x=all_scores,
            y=pos_survival,
            mode='lines',
            name='Positive Class',
            line=dict(color='#FF6B6B', width=4),
            hovertemplate=(
                    'Score: %{x:.3f}<br>' +
                    '% Positive with score ≥ x: %{y:.1%}<br>' +
                    '<extra>Positive Class</extra>'
            )
        ),
        row=2, col=1
    )

    # Negative class survival function
    fig.add_trace(
        go.Scatter(
            x=all_scores,
            y=neg_survival,
            mode='lines',
            name='Negative Class',
            line=dict(color='#4ECDC4', width=4),
            hovertemplate=(
                    'Score: %{x:.3f}<br>' +
                    '% Negative with score ≥ x: %{y:.1%}<br>' +
                    '<extra>Negative Class</extra>'
            )
        ),
        row=2, col=1
    )

    # KS difference area (shaded between curves)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([all_scores, all_scores[::-1]]),
            y=np.concatenate([pos_survival, neg_survival[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255, 165, 0, 0)'),
            name='KS Difference Area',
            hovertemplate='<extra>KS Area</extra>',
            showlegend=True
        ),
        row=2, col=1
    )

    # KS maximum point
    fig.add_trace(
        go.Scatter(
            x=[ks_threshold],
            y=[pos_survival[ks_idx]],
            mode='markers+text',
            marker=dict(
                color='red',
                size=15,
                symbol='diamond',
                line=dict(color='black', width=2)
            ),
            text=[f'KS={ks_statistic:.3f}'],
            textposition='top center',
            textfont=dict(size=14, color='red'),
            name='KS Maximum',
            hovertemplate=(
                    f'Threshold: {ks_threshold:.3f}<br>' +
                    f'KS: {ks_statistic:.3f}<br>' +
                    f'% Positive ≥ threshold: {pos_survival[ks_idx]:.1%}<br>' +
                    f'% Negative ≥ threshold: {neg_survival[ks_idx]:.1%}<br>' +
                    '<extra>KS Maximum</extra>'
            ),
            showlegend=False
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=700,
        width=700,
        title=dict(
            text=f"<b>Kolmogorov-Smirnov Analysis: {model_name}</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        showlegend=True,
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
        hovermode='x unified'
    )

    # Update axes
    fig.update_xaxes(
        title_text="<b>Predicted Score / Probability</b>",
        row=1, col=1,
        range=[score_min - 0.02, score_max + 0.02]
    )
    fig.update_xaxes(
        title_text="<b>Predicted Score / Probability</b>",
        row=2, col=1,
        range=[score_min - 0.02, score_max + 0.02]
    )
    fig.update_yaxes(
        title_text="<b>Density</b>",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="<b>Proportion with Score ≥ X</b>",
        row=2, col=1,
        range=[-0.02, 1.02],
        tickformat='.0%',
        gridcolor='lightgray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=1
    )

    # Add KS interpretation
    if ks_statistic >= 0.75:
        interpretation = "EXCELLENT discrimination"
        color = "#27AE60"  # Green
    elif ks_statistic >= 0.6:
        interpretation = "VERY GOOD discrimination"
        color = "#2ECC71"  # Light green
    elif ks_statistic >= 0.4:
        interpretation = "GOOD discrimination"
        color = "#F1C40F"  # Yellow
    elif ks_statistic >= 0.2:
        interpretation = "FAIR discrimination"
        color = "#E67E22"  # Orange
    else:
        interpretation = "POOR discrimination"
        color = "#E74C3C"  # Red

    fig.add_annotation(
        x=0.98,
        y=0.45,
        xref="paper",
        yref="paper",
        text=(
            f"<b>KS = {ks_statistic:.3f}</b><br>"
            f"<span style='color:{color}'><b>{interpretation}</b></span><br><br>"
            f"• Higher KS = Better model<br>"
            f"• Max at score = {ks_threshold:.3f}<br>"
            f"• Positive ≥ threshold: {pos_survival[ks_idx]:.1%}<br>"
            f"• Negative ≥ threshold: {neg_survival[ks_idx]:.1%}"
        ),
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor=color,
        borderwidth=2,
        borderpad=8
    )

    # Add grid lines for better readability
    for i in np.arange(0.1, 1.0, 0.1):
        fig.add_hline(
            y=i,
            line_dash="dot",
            line_color="lightgray",
            line_width=1,
            row=2, col=1
        )

    return fig


def get_interactive_ks_simple(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Simple, clean KS plot showing survival functions.
    This is the most commonly used KS visualization.
    """
    import numpy as np

    # Separate scores by class
    pos_scores = y_pred_proba[y_test == 1]
    neg_scores = y_pred_proba[y_test == 0]

    # Create smooth grid of scores
    score_min = min(pos_scores.min(), neg_scores.min())
    score_max = max(pos_scores.max(), neg_scores.max())
    all_scores = np.linspace(score_min, score_max, 1000)

    # Calculate survival functions
    def survival_function(x, data):
        return np.mean(data >= x)

    pos_survival = np.array([survival_function(s, pos_scores) for s in all_scores])
    neg_survival = np.array([survival_function(s, neg_scores) for s in all_scores])

    # Calculate KS statistic
    ks_diff = pos_survival - neg_survival
    ks_statistic = np.max(np.abs(ks_diff))
    ks_idx = np.argmax(np.abs(ks_diff))
    ks_threshold = all_scores[ks_idx]

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # 1. Positive class survival function
    fig.add_trace(go.Scatter(
        x=all_scores,
        y=pos_survival,
        mode='lines',
        name=f'Positive Class',
        line=dict(
            color='#FF6B6B',
            width=4
        ),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.1)',
        hovertemplate=(
                'Score: %{x:.3f}<br>' +
                '% Positive ≥ score: %{y:.1%}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 2. Negative class survival function
    fig.add_trace(go.Scatter(
        x=all_scores,
        y=neg_survival,
        mode='lines',
        name=f'Negative Class',
        line=dict(
            color='#4ECDC4',
            width=4
        ),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.1)',
        hovertemplate=(
                'Score: %{x:.3f}<br>' +
                '% Negative ≥ score: %{y:.1%}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 3. KS difference line
    fig.add_trace(go.Scatter(
        x=all_scores,
        y=ks_diff,
        mode='lines',
        name=f'KS Difference (max = {ks_statistic:.3f})',
        line=dict(
            color='#FFA726',
            width=3,
            dash='dash'
        ),
        hovertemplate=(
                'Score: %{x:.3f}<br>' +
                'KS Difference: %{y:.3f}<br>' +
                '<extra></extra>'
        ),
        showlegend=True
    ))

    # 4. KS maximum point
    fig.add_trace(go.Scatter(
        x=[ks_threshold],
        y=[ks_statistic],
        mode='markers+text',
        marker=dict(
            color='#D32F2F',
            size=15,
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        text=[f'KS={ks_statistic:.3f}'],
        textposition='top center',
        textfont=dict(size=14, color='#D32F2F'),
        name='KS Maximum',
        hovertemplate=(
                f'Threshold: {ks_threshold:.3f}<br>' +
                f'KS: {ks_statistic:.3f}<br>' +
                f'% Positive ≥ threshold: {pos_survival[ks_idx]:.1%}<br>' +
                f'% Negative ≥ threshold: {neg_survival[ks_idx]:.1%}<br>' +
                '<extra></extra>'
        ),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        width=700,
        height=700,
        title=dict(
            text=f"<b>Kolmogorov-Smirnov Curve (KS = {ks_statistic:.3f}) - {model_name}</b>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Predicted Score / Probability</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[score_min - 0.02, score_max + 0.02],
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text="<b>Proportion with Score ≥ X</b>",
                font=dict(size=16, color="#34495E")
            ),
            range=[-0.02, 1.02],
            tickformat='.0%',
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
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
            font=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=60, b=8),
        hovermode='x unified'
    )

    # Add interpretation guide
    ks_guide = [
        (0.0, 0.2, "❌ Poor", "#E74C3C"),
        (0.2, 0.4, "⚠️ Fair", "#F39C12"),
        (0.4, 0.6, "✅ Good", "#2ECC71"),
        (0.6, 0.8, "✨ Very Good", "#27AE60"),
        (0.8, 1.0, "🏆 Excellent", "#1D8348")
    ]

    # Find which range KS falls into
    ks_category = ""
    ks_color = "#000000"
    for low, high, category, color in ks_guide:
        if low <= ks_statistic < high:
            ks_category = category
            ks_color = color
            break

    # Add annotation
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            f"<b>KS = {ks_statistic:.3f}</b><br>"
            f"<span style='color:{ks_color}'><b>{ks_category}</b></span><br><br>"
            f"At threshold = {ks_threshold:.3f}:<br>"
            f"• {pos_survival[ks_idx]:.1%} of positives<br>"
            f"• {neg_survival[ks_idx]:.1%} of negatives<br>"
            f"have score ≥ threshold"
        ),
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor=ks_color,
        borderwidth=2,
        borderpad=8
    )

    return fig