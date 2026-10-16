"""
visualizations.py - Interactive Plotly charts for ChurnIQ dashboard.
Uses a refined blue/teal/coral business-analytics palette with consistent
typography, grid styling, and hover interactions across all charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Refined Color Tokens ---
COLOR_PRIMARY = "#4f8b7e"
COLOR_ACCENT = "#2c5d53"
COLOR_TEAL = "#4f8b7e"
COLOR_CHURN = "#c25953"
COLOR_RETAIN = "#4f8b7e"
COLOR_NEUTRAL = "#7a766d"
PALETTE_MODELS = ["#4f8b7e", "#2c5d53", "#70a99d", "#d97736", "#c25953"]

# --- Shared Layout Template ---
_LAYOUT_DEFAULTS = dict(
    font=dict(family="sans serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13, color="#1c2b2b"),
    title_font=dict(size=16, color="#1c2b2b", family="sans serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="#1c2b2b", font_size=12, font_color="#fbfaf6", bordercolor="#4f8b7e"),
    margin=dict(t=44, b=28, l=28, r=20),
)


def _apply_defaults(fig, **overrides):
    """Apply shared layout defaults plus any per-chart overrides."""
    merged = {**_LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**merged)
    # Grid styling for cartesian charts
    fig.update_xaxes(
        gridcolor="#e5e1d5",
        gridwidth=0.5,
        zerolinecolor="#ccc7ba",
        tickfont=dict(size=11, color="#7a766d"),
        title_font=dict(size=12, color="#1c2b2b"),
    )
    fig.update_yaxes(
        gridcolor="#e5e1d5",
        gridwidth=0.5,
        zerolinecolor="#ccc7ba",
        tickfont=dict(size=11, color="#7a766d"),
        title_font=dict(size=12, color="#1c2b2b"),
    )
    return fig


def plot_churn_distribution(df: pd.DataFrame, col: str = "Churn", title: str = "Customer Churn Breakdown"):
    """Donut chart showing overall churn vs retained distribution."""
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    colors = [COLOR_CHURN if str(s) in ["Yes", "1"] else COLOR_RETAIN for s in counts["Status"]]

    fig = go.Figure(data=[go.Pie(
        labels=counts["Status"],
        values=counts["Count"],
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13, color="#1e293b"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    )])

    fig = _apply_defaults(fig,
        title=f"<b>{title}</b>",
        margin=dict(t=44, b=12, l=12, r=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        height=320,
        showlegend=True,
    )
    return fig


def plot_segment_churn(df: pd.DataFrame, segment_col: str, title: str = None):
    """Horizontal bar chart showing churn percentage across customer segments."""
    grouped = df.groupby(segment_col)["Churn"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).reset_index()
    grouped.columns = [segment_col, "Churn_Rate"]
    grouped = grouped.sort_values("Churn_Rate", ascending=True)

    fig = go.Figure(go.Bar(
        x=grouped["Churn_Rate"],
        y=grouped[segment_col].astype(str),
        orientation="h",
        marker=dict(
            color=grouped["Churn_Rate"],
            colorscale=[[0, "#16a34a"], [0.5, "#0ea5e9"], [1, "#dc2626"]],
            showscale=False,
            line=dict(color="#ffffff", width=0.5),
        ),
        text=[f"{v:.1f}%" for v in grouped["Churn_Rate"]],
        textposition="outside",
        textfont=dict(size=12, color="#475569"),
        hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.2f}%<extra></extra>",
        width=0.6,
    ))

    fig = _apply_defaults(fig,
        title=f"<b>{title or f'Churn Rate by {segment_col}'}</b>",
        xaxis=dict(title="Churn Rate (%)", range=[0, max(grouped["Churn_Rate"]) * 1.28]),
        yaxis=dict(title="", tickfont=dict(size=12, color="#334155")),
        margin=dict(t=44, b=28, l=130, r=30),
        height=300,
    )
    return fig


def plot_model_comparison_bar(metrics_df: pd.DataFrame, metric: str = "ROC-AUC"):
    """Ranks and compares all evaluated models across selected metric."""
    df_sorted = metrics_df.sort_values(metric, ascending=True).reset_index()

    fig = go.Figure(go.Bar(
        x=df_sorted[metric],
        y=df_sorted["Model"],
        orientation="h",
        marker=dict(
            color=[COLOR_CHURN if m == "CatBoost" else COLOR_ACCENT for m in df_sorted["Model"]],
            line=dict(color="#ffffff", width=0.5),
        ),
        text=[f"{v:.4f}" for v in df_sorted[metric]],
        textposition="outside",
        textfont=dict(size=12, color="#475569"),
        hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>",
        width=0.55,
    ))

    fig = _apply_defaults(fig,
        title=f"<b>Model Performance Comparison — {metric}</b>",
        xaxis=dict(title=metric, range=[0, 1.06]),
        yaxis=dict(title="", tickfont=dict(size=12, color="#334155")),
        margin=dict(t=44, b=28, l=160, r=50),
        height=320,
    )
    return fig


def plot_multi_roc_curves(test_eval_cache: dict):
    """Overlaid ROC curves for all models evaluated in research."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#cbd5e1", dash="dash", width=1.5),
        name="Random (AUC = 0.50)",
    ))

    models = test_eval_cache.get("models", {})
    colors = PALETTE_MODELS

    for idx, (name, data) in enumerate(models.items()):
        fpr = data["fpr"]
        tpr = data["tpr"]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.8 if "CatBoost" in name else 2),
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
        ))

    fig = _apply_defaults(fig,
        title="<b>ROC Curves — All Models</b>",
        xaxis=dict(title="False Positive Rate (1 − Specificity)", range=[-0.01, 1.01]),
        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.01, 1.02]),
        margin=dict(t=44, b=44, l=44, r=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    return fig


def plot_multi_pr_curves(test_eval_cache: dict):
    """Overlaid Precision-Recall curves for all models."""
    fig = go.Figure()

    models = test_eval_cache.get("models", {})
    colors = PALETTE_MODELS

    for idx, (name, data) in enumerate(models.items()):
        prec = data["prec_curve"]
        rec = data["rec_curve"]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=rec, y=prec,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.8 if "CatBoost" in name else 2),
            hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
        ))

    fig = _apply_defaults(fig,
        title="<b>Precision-Recall Curves — All Models</b>",
        xaxis=dict(title="Recall", range=[-0.01, 1.01]),
        yaxis=dict(title="Precision", range=[-0.01, 1.02]),
        margin=dict(t=44, b=44, l=44, r=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    return fig


def plot_confusion_matrix_heatmap(cm: list, model_name: str = "CatBoost"):
    """Normalized & counts annotated confusion matrix."""
    cm_arr = np.array(cm)
    total = np.sum(cm_arr)
    cm_norm = (cm_arr / total) * 100

    labels = ["Retained (0)", "Churned (1)"]
    annotations = []
    for r in range(2):
        for c in range(2):
            count = cm_arr[r, c]
            pct = cm_norm[r, c]
            annotations.append(dict(
                x=labels[c],
                y=labels[r],
                text=f"<b>{count:,}</b><br>({pct:.1f}%)",
                showarrow=False,
                font=dict(color="white" if pct > 35 else "#1e293b", size=14),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        colorscale=[[0, "#f1f5f9"], [0.4, "#0ea5e9"], [1, "#1e3a8a"]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:.1f}%<extra></extra>",
    ))

    fig = _apply_defaults(fig,
        title=f"<b>Confusion Matrix — {model_name}</b><br><sub style='font-size:11px;color:#64748b'>Test Set, N={total:,}</sub>",
        xaxis=dict(title="Predicted Outcome", side="bottom"),
        yaxis=dict(title="Actual Ground Truth", autorange="reversed"),
        annotations=annotations,
        margin=dict(t=58, b=36, l=80, r=40),
        height=360,
    )
    return fig


def plot_feature_importance_bar(df_imp: pd.DataFrame, model_name: str = "CatBoost"):
    """Ranked global feature importance chart."""
    df_sorted = df_imp.sort_values("Normalized_Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["Normalized_Importance"] * 100,
        y=df_sorted["Clean_Feature"],
        orientation="h",
        marker=dict(color=COLOR_ACCENT, line=dict(color="#ffffff", width=0.5)),
        text=[f"{v * 100:.1f}%" for v in df_sorted["Normalized_Importance"]],
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
        hovertemplate="<b>%{y}</b><br>Relative Importance: %{x:.2f}%<extra></extra>",
        width=0.6,
    ))

    fig = _apply_defaults(fig,
        title=f"<b>Global Feature Importance — {model_name}</b>",
        xaxis=dict(title="Normalized Importance (%)", range=[0, max(df_sorted["Normalized_Importance"] * 100) * 1.28]),
        yaxis=dict(title="", tickfont=dict(size=12, color="#334155")),
        margin=dict(t=44, b=28, l=170, r=50),
        height=450,
    )
    return fig


def plot_causal_quadrant_scatter(eval_df: pd.DataFrame):
    """Scatter plot of Churn Probability Without Offer vs Churn Probability With Offer."""
    quadrant_colors = {
        "Persuadable (Prime Target)": "#16a34a",
        "Lost Cause (Avoid High Cost)": "#dc2626",
        "Sure Thing (Organic Retention)": "#0ea5e9",
        "Sleeping Dog (Do Not Disturb)": "#7c2d12",
    }

    fig = go.Figure()

    for segment, color in quadrant_colors.items():
        sub = eval_df[eval_df["Causal_Segment"] == segment]
        if len(sub) > 0:
            fig.add_trace(go.Scatter(
                x=sub["Predicted_Churn_No_Offer"],
                y=sub["Predicted_Churn_With_Offer"],
                mode="markers",
                name=f"{segment} ({len(sub):,})",
                marker=dict(size=6, color=color, opacity=0.65, line=dict(width=0)),
                hovertemplate="<b>" + segment + "</b><br>P(No Offer): %{x:.2%}<br>P(With Offer): %{y:.2%}<extra></extra>"
            ))

    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, line=dict(color="#cbd5e1", dash="dot", width=1.2))
    fig.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5, line=dict(color="#cbd5e1", dash="dot", width=1.2))

    fig = _apply_defaults(fig,
        title="<b>Causal Uplift Quadrants</b>",
        xaxis=dict(title="Predicted Churn Risk — No Offer (Control)", range=[-0.02, 1.02]),
        yaxis=dict(title="Predicted Churn Risk — With Offer (Treated)", range=[-0.02, 1.02]),
        margin=dict(t=44, b=44, l=44, r=20),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    return fig


def plot_uplift_rank_deciles(eval_df: pd.DataFrame):
    """Cumulative uplift / decile performance curve comparing T-Learner targeting vs Random Benchmark."""
    eval_df = eval_df.sort_values("Estimated_Uplift", ascending=False).reset_index(drop=True)
    eval_df["Decile"] = pd.qcut(eval_df.index, q=10, labels=[f"D{i+1}" for i in range(10)])

    decile_stats = eval_df.groupby("Decile", observed=False).agg({
        "True_Uplift": "mean",
        "Estimated_Uplift": "mean"
    }).reset_index()

    overall_true = eval_df["True_Uplift"].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decile_stats["Decile"],
        y=decile_stats["True_Uplift"] * 100,
        name="Ground-Truth Uplift per Decile",
        marker=dict(color=COLOR_TEAL, line=dict(color="#ffffff", width=0.5)),
        text=[f"{v*100:.1f}%" for v in decile_stats["True_Uplift"]],
        textposition="outside",
        textfont=dict(size=10, color="#475569"),
        hovertemplate="<b>%{x}</b><br>True Uplift: %{y:.2f}%<extra></extra>",
        width=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=decile_stats["Decile"],
        y=[overall_true * 100] * 10,
        mode="lines+markers",
        name=f"Random Benchmark ({overall_true * 100:.2f}%)",
        line=dict(color=COLOR_CHURN, dash="dash", width=2),
        marker=dict(size=5),
        hovertemplate="Random Benchmark: %{y:.2f}%<extra></extra>",
    ))

    fig = _apply_defaults(fig,
        title="<b>T-Learner Targeting Effectiveness — Uplift by Decile</b>",
        xaxis=dict(title="Uplift Deciles (D1 = Highest Predicted Treatment Benefit)"),
        yaxis=dict(title="True Churn Reduction (%)"),
        margin=dict(t=44, b=40, l=44, r=20),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="center", x=0.5),
    )
    return fig
