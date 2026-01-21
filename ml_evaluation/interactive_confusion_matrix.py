import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_interactive_confusion_matrix(y_test, y_pred, y_pred_proba=None, model_name: str = "",
                                     threshold: float = 0.5) -> go.Figure:
    """
    Create interactive confusion matrix with multiple visualizations.

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like or None
        Predicted labels (if None, will use y_pred_proba > threshold)
    y_pred_proba : array-like, optional
        Predicted probabilities (for threshold slider)
    model_name : str
        Name of the model for title
    threshold : float
        Classification threshold (if y_pred is None)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    import pandas as pd

    # Convert predictions if needed
    if y_pred is None and y_pred_proba is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    elif y_pred is None:
        raise ValueError("Either y_pred or y_pred_proba must be provided")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Confusion Matrix (Counts)</b>",
            "<b>Confusion Matrix (Normalized)</b>",
            "<b>Classification Metrics</b>",
            "<b>Error Analysis</b>"
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # ============ TOP-LEFT: Confusion Matrix (Counts) ============
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale=[[0, '#E8F8F5'], [0.5, '#82E0AA'], [1, '#27AE60']],
            showscale=True,
            colorbar=dict(title="Count", len=0.4, y=0.8),
            hovertemplate=(
                    'Actual: %{y}<br>' +
                    'Predicted: %{x}<br>' +
                    'Count: %{z}<br>' +
                    '<extra></extra>'
            )
        ),
        row=1, col=1
    )

    # ============ TOP-RIGHT: Confusion Matrix (Normalized) ============
    fig.add_trace(
        go.Heatmap(
            z=cm_normalized,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=[['{:.1%}'.format(val) for val in row] for row in cm_normalized],
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale=[[0, '#EBF5FB'], [0.5, '#5DADE2'], [1, '#2E86C1']],
            showscale=True,
            colorbar=dict(title="Percentage", len=0.4, y=0.8, tickformat='.0%'),
            hovertemplate=(
                    'Actual: %{y}<br>' +
                    'Predicted: %{x}<br>' +
                    'Percentage: %{z:.1%}<br>' +
                    '<extra></extra>'
            )
        ),
        row=1, col=2
    )

    # ============ BOTTOM-LEFT: Classification Metrics ============
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score']
    values = [accuracy, precision, recall, specificity, f1]
    colors = ['#27AE60', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12']

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=1,
            opacity=0.8,
            text=[f'{v:.1%}' for v in values],
            textposition='auto',
            textfont=dict(size=12, color='white', weight='bold'),
            hovertemplate=(
                    'Metric: %{x}<br>' +
                    'Value: %{y:.1%}<br>' +
                    '<extra></extra>'
            )
        ),
        row=2, col=1
    )

    # ============ BOTTOM-RIGHT: Error Analysis ============
    error_types = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
    error_counts = [tp, tn, fp, fn]
    error_colors = ['#27AE60', '#27AE60', '#E74C3C', '#E74C3C']

    fig.add_trace(
        go.Bar(
            x=error_types,
            y=error_counts,
            marker_color=error_colors,
            marker_line_color='white',
            marker_line_width=1,
            opacity=0.8,
            text=error_counts,
            textposition='auto',
            textfont=dict(size=14, color='white', weight='bold'),
            hovertemplate=(
                    'Type: %{x}<br>' +
                    'Count: %{y}<br>' +
                    '<extra></extra>'
            )
        ),
        row=2, col=2
    )

    # ============ UPDATE LAYOUT ============
    fig.update_layout(
        height=900,
        width=1200,
        title=dict(
            text=f"<b>Confusion Matrix Analysis: {model_name}</b><br>"
                 f"<span style='font-size:14px; color:#7F8C8D'>Threshold = {threshold:.2f} | Total Samples = {len(y_test):,}</span>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=100, b=8)
    )

    # Update axes
    fig.update_xaxes(title_text="<b>Predicted Label</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Actual Label</b>", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title_text="<b>Predicted Label</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Actual Label</b>", row=1, col=2, autorange="reversed")
    fig.update_xaxes(title_text="<b>Metric</b>", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="<b>Value</b>", row=2, col=1, tickformat='.0%')
    fig.update_xaxes(title_text="<b>Classification Result</b>", row=2, col=2)
    fig.update_yaxes(title_text="<b>Count</b>", row=2, col=2)

    # Add summary statistics
    summary_text = (
        f"<b>Model Performance Summary:</b><br>"
        f"• Accuracy: <span style='color:#27AE60'>{accuracy:.1%}</span><br>"
        f"• Precision: <span style='color:#3498DB'>{precision:.1%}</span><br>"
        f"• Recall: <span style='color:#9B59B6'>{recall:.1%}</span><br>"
        f"• F1-Score: <span style='color:#F39C12'>{f1:.1%}</span><br>"
        f"• Specificity: <span style='color:#E74C3C'>{specificity:.1%}</span>"
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=summary_text,
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#2C3E50",
        borderwidth=2,
        borderpad=8
    )

    # Add interpretation
    if f1 >= 0.9:
        rating = "🏆 EXCELLENT"
        color = "#27AE60"
    elif f1 >= 0.8:
        rating = "✅ VERY GOOD"
        color = "#2ECC71"
    elif f1 >= 0.7:
        rating = "👍 GOOD"
        color = "#F1C40F"
    elif f1 >= 0.6:
        rating = "⚠️ FAIR"
        color = "#E67E22"
    else:
        rating = "❌ NEEDS IMPROVEMENT"
        color = "#E74C3C"

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<span style='color:{color}'><b>{rating}</b></span>",
        showarrow=False,
        font=dict(size=16, weight='bold'),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor=color,
        borderwidth=2,
        borderpad=10
    )

    return fig


def get_interactive_confusion_matrix_with_slider(y_test, y_pred_proba, model_name: str = "") -> go.Figure:
    """
    Interactive confusion matrix with THRESHOLD SLIDER to explore different cutoffs.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    # Create initial confusion matrix with threshold=0.5
    y_pred_initial = (y_pred_proba >= 0.5).astype(int)
    cm_initial = confusion_matrix(y_test, y_pred_initial)

    # Create figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)

    # Add initial heatmap
    fig.add_trace(go.Heatmap(
        z=cm_initial,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        text=cm_initial,
        texttemplate='%{text}',
        textfont={"size": 24},
        colorscale=[[0, '#E8F8F5'], [1, '#27AE60']],
        colorbar=dict(title="Count"),
        hovertemplate=(
                'Actual: %{y}<br>' +
                'Predicted: %{x}<br>' +
                'Count: %{z}<br>' +
                '<extra></extra>'
        )
    ))

    # Calculate metrics for different thresholds (for slider steps)
    thresholds = np.linspace(0, 1, 101)

    # Create slider steps
    steps = []
    for threshold in thresholds[::10]:  # Every 10th threshold for performance
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        cm_temp = confusion_matrix(y_test, y_pred_temp)

        step = dict(
            method="update",
            args=[{
                "z": [cm_temp],
                "text": [cm_temp]
            }],
            label=f"{threshold:.1f}"
        )
        steps.append(step)

    # Update layout
    fig.update_layout(
        height=700,
        width=700,
        title=dict(
            text=f"<b>Interactive Confusion Matrix: {model_name}</b><br>"
                 f"<span style='font-size:14px; color:#7F8C8D'>Drag slider to adjust classification threshold</span>",
            font=dict(family="Arial", size=24, color="#2C3E50"),
            x=0.5,
            xanchor="center"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=120, b=80),
        xaxis_title="<b>Predicted Label</b>",
        yaxis_title="<b>Actual Label</b>",
        yaxis_autorange="reversed"
    )

    # Add slider (FIXED: no id parameter)
    sliders = [dict(
        active=50,  # Default to 0.5 threshold
        currentvalue={"prefix": "Threshold: ", "font": {"size": 14}},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)

    # Add metrics display (FIXED: removed id parameter)
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            "<b>Metrics at current threshold:</b><br>"
            "• Accuracy: --<br>"
            "• Precision: --<br>"
            "• Recall: --<br>"
            "• F1-Score: --"
        ),
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#2C3E50",
        borderwidth=2,
        borderpad=8
    )

    return fig