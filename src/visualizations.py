"""
visualizations.py - Interactive Plotly charts for ChurnIQ dashboard.
Adheres to the restrained teal/blue and deep navy business-analytics aesthetic.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Color tokens
COLOR_PRIMARY = "#1d3557"
COLOR_ACCENT = "#457b9d"
COLOR_TEAL = "#2a9d8f"
COLOR_CHURN = "#e63946"
COLOR_RETAIN = "#2a9d8f"
COLOR_NEUTRAL = "#6c757d"
PALETTE_MODELS = ["#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#9d0208"]


def plot_churn_distribution(df: pd.DataFrame, col: str = "Churn", title: str = "Customer Churn Breakdown"):
    """Donut chart showing overall churn vs retained distribution."""
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    colors = [COLOR_CHURN if str(s) in ["Yes", "1"] else COLOR_RETAIN for s in counts["Status"]]

    fig = go.Figure(data=[go.Pie(
        labels=counts["Status"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title=f"<b>{title}</b>",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
            colorscale=[[0, "#2a9d8f"], [0.5, "#457b9d"], [1, "#e63946"]],
            showscale=False
        ),
        text=[f"{v:.1f}%" for v in grouped["Churn_Rate"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>{title or f'Churn Rate by {segment_col}'}</b>",
        xaxis=dict(title="Churn Rate (%)", range=[0, max(grouped["Churn_Rate"]) * 1.25]),
        yaxis=dict(title=""),
        margin=dict(t=40, b=30, l=120, r=30),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
            color=["#e63946" if m == "catboost_model" or m == "CatBoost" else "#457b9d" for m in df_sorted["Model"]]
        ),
        text=[f"{v:.4f}" for v in df_sorted[metric]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Model Performance Comparison ({metric})</b>",
        xaxis=dict(title=metric, range=[0, 1.05]),
        yaxis=dict(title=""),
        margin=dict(t=40, b=30, l=150, r=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_multi_roc_curves(test_eval_cache: dict):
    """Overlaid ROC curves for all models evaluated in research."""
    fig = go.Figure()

    # Diagonal random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#adb5bd", dash="dash", width=1.5),
        name="Random Chance (AUC = 0.50)"
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
            line=dict(color=color, width=2.5 if "CatBoost" in name else 1.8),
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Curves</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.01, 1.01]),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[-0.01, 1.02]),
        margin=dict(t=40, b=40, l=40, r=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
            line=dict(color=color, width=2.5 if "CatBoost" in name else 1.8),
            hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        title="<b>Precision-Recall (PR) Curves</b>",
        xaxis=dict(title="Recall", range=[-0.01, 1.01]),
        yaxis=dict(title="Precision", range=[-0.01, 1.02]),
        margin=dict(t=40, b=40, l=40, r=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
                font=dict(color="white" if pct > 35 else "#1d3557", size=14)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        colorscale=[[0, "#f8f9fa"], [0.5, "#457b9d"], [1, "#1d3557"]],
        showscale=False
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix — {model_name} (Test Set, N={total:,})</b>",
        xaxis=dict(title="Predicted Outcome", side="bottom"),
        yaxis=dict(title="Actual Ground Truth", autorange="reversed"),
        annotations=annotations,
        margin=dict(t=50, b=40, l=80, r=40),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_feature_importance_bar(df_imp: pd.DataFrame, model_name: str = "CatBoost"):
    """Ranked global feature importance chart."""
    df_sorted = df_imp.sort_values("Normalized_Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["Normalized_Importance"] * 100,
        y=df_sorted["Clean_Feature"],
        orientation="h",
        marker=dict(color="#457b9d"),
        text=[f"{v * 100:.1f}%" for v in df_sorted["Normalized_Importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Relative Importance: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Global Feature Importance ({model_name})</b>",
        xaxis=dict(title="Normalized Importance (%)", range=[0, max(df_sorted["Normalized_Importance"] * 100) * 1.25]),
        yaxis=dict(title=""),
        margin=dict(t=40, b=30, l=160, r=40),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_causal_quadrant_scatter(eval_df: pd.DataFrame):
    """
    Scatter plot of Churn Probability Without Offer vs Churn Probability With Offer.
    Visualizes the four causal quadrants in uplift modeling.
    """
    quadrant_colors = {
        "Persuadable (Prime Target)": "#2a9d8f",
        "Lost Cause (Avoid High Cost)": "#e63946",
        "Sure Thing (Organic Retention)": "#457b9d",
        "Sleeping Dog (Do Not Disturb)": "#9d0208"
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
                marker=dict(size=7, color=color, opacity=0.7),
                hovertemplate="<b>" + segment + "</b><br>P(No Offer): %{x:.2%}<br>P(With Offer): %{y:.2%}<extra></extra>"
            ))

    # Reference lines at 0.5 threshold
    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, line=dict(color="#adb5bd", dash="dot"))
    fig.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5, line=dict(color="#adb5bd", dash="dot"))

    fig.update_layout(
        title="<b>Causal Uplift Quadrants: Effect of Retention Intervention</b>",
        xaxis=dict(title="Predicted Churn Risk WITHOUT Offer (P_control)", range=[-0.02, 1.02]),
        yaxis=dict(title="Predicted Churn Risk WITH Offer (P_treated)", range=[-0.02, 1.02]),
        margin=dict(t=40, b=40, l=40, r=20),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_uplift_rank_deciles(eval_df: pd.DataFrame):
    """
    Cumulative uplift / decile performance curve comparing T-Learner targeting vs Random Benchmark.
    """
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
        name="Mean Ground-Truth Uplift in Decile (%)",
        marker=dict(color="#2a9d8f")
    ))
    fig.add_trace(go.Scatter(
        x=decile_stats["Decile"],
        y=[overall_true * 100] * 10,
        mode="lines",
        name=f"Random Benchmark ({overall_true * 100:.2f}%)",
        line=dict(color="#e63946", dash="dash", width=2)
    ))

    fig.update_layout(
        title="<b>T-Learner Targeting Effectiveness: Uplift by Decile</b>",
        xaxis=dict(title="Uplift Deciles (D1 = Highest Predicted Treatment Benefit)"),
        yaxis=dict(title="True Churn Reduction Benefit (%)"),
        margin=dict(t=40, b=40, l=40, r=20),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
