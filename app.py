"""
app.py - ChurnIQ: Explainable Customer Intelligence Dashboard.
A research-grade interactive analytics application for customer churn prediction,
explainable AI, counterfactual what-if analysis, and causal uplift decision support.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data_loader import load_telco_data, load_synthetic_uplift_data
from src.preprocessing import (
    engineer_features,
    prepare_churn_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLS
)
from src.models_manager import load_artifacts, train_and_save_artifacts
from src.prediction import predict_single_customer, predict_batch, categorize_risk
from src.explainability import get_global_feature_importance, explain_single_customer
from src.counterfactuals import (
    init_dice_explainer,
    generate_dice_counterfactuals,
    simulate_business_what_if_scenarios,
    ACTIONABLE_FEATURES
)
from src.uplift import train_or_load_t_learner
import src.visualizations as viz

# Set Streamlit page configuration
st.set_page_config(
    page_title="ChurnIQ — Explainable Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Analytics Dashboard CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #f3f1e9;
        color: #1c2b2b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #162423 0%, #1e3130 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #fbfaf6 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2c4240 !important;
        opacity: 0.6;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #a4beba !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        padding: 6px 10px !important;
        border-radius: 6px;
        transition: background-color 0.2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: rgba(255,255,255,0.08);
    }

    /* --- KPI Cards --- */
    .kpi-card {
        background: #fbfaf6;
        border: 1px solid #e5e1d5;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 12px rgba(0,0,0,0.02);
        margin-bottom: 12px;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 12px 0 0 12px;
    }
    .kpi-card:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.05), 0 10px 28px rgba(0,0,0,0.04);
        transform: translateY(-1px);
    }
    .kpi-card.accent-primary::before { background: #4f8b7e; }
    .kpi-card.accent-warning::before { background: #d97736; }
    .kpi-card.accent-danger::before { background: #c25953; }
    .kpi-card.accent-success::before { background: #4f8b7e; }
    .kpi-card.accent-neutral::before { background: #7a766d; }

    .kpi-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6b665c;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1c2b2b;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #7a766d;
        font-weight: 400;
    }

    /* --- Status Badges --- */
    .badge-high {
        background: #fef2f2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.76rem;
        border: 1px solid #fecaca;
        display: inline-block;
    }
    .badge-medium {
        background: #fffbeb;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.76rem;
        border: 1px solid #fde68a;
        display: inline-block;
    }
    .badge-low {
        background: #f0fdf4;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.76rem;
        border: 1px solid #bbf7d0;
        display: inline-block;
    }

    /* --- Callouts --- */
    .info-callout {
        background: #eff6ff;
        border-left: 3px solid #2563eb;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #1e3a5f;
        font-weight: 400;
    }
    .warning-callout {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #78350f;
        font-weight: 400;
    }

    /* --- Section headers --- */
    .stApp h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #0f172a !important;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 8px !important;
    }
    .stApp h2, .stApp h3 {
        font-weight: 600 !important;
        color: #1e293b !important;
        letter-spacing: -0.01em;
    }

    /* --- Sidebar brand --- */
    .sidebar-brand {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .sidebar-sub {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 400;
        margin-top: -2px;
    }

    /* --- Divider --- */
    .section-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 18px 0;
    }

    /* --- Recommendation cards --- */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: box-shadow 0.2s ease;
    }
    .rec-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .rec-priority-tag {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .rec-priority-high { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .rec-priority-medium { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .rec-priority-low { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

    /* --- Scenario rows --- */
    .scenario-row {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .scenario-row:hover {
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transform: translateY(-1px);
    }

    /* --- Data table polish --- */
    .stDataFrame [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* --- Tab styling --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(37, 99, 235, 0.06);
    }

    /* --- Expander polish --- */
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }

    /* --- Button hover --- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* --- st.metric polish --- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* --- Subtle fade-in --- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .kpi-card, .rec-card, .scenario-row, .profile-card, .factor-card {
        animation: fadeInUp 0.35s ease-out;
    }

    /* --- Profile card --- */
    .profile-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 12px;
    }
    .profile-card .profile-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .profile-card .profile-value {
        font-size: 0.92rem;
        font-weight: 500;
        color: #1e293b;
        margin-bottom: 10px;
    }

    /* --- Factor cards (XAI) --- */
    .factor-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .factor-card.risk { border-left: 3px solid #dc2626; }
    .factor-card.protective { border-left: 3px solid #16a34a; }
    .factor-card .factor-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .factor-card .factor-value {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 6px;
    }
    .factor-card .factor-desc {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
    }

    /* --- Page subtitle --- */
    .page-subtitle {
        font-size: 0.92rem;
        color: #64748b;
        font-weight: 400;
        margin-top: -4px;
        margin-bottom: 16px;
    }

    /* --- Pipeline diagram --- */
    .pipeline-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        font-family: 'Inter', monospace;
        font-size: 0.82rem;
        color: #334155;
        overflow-x: auto;
        line-height: 1.5;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_telco_data():
    """Cached loader for raw and feature-engineered Telco data."""
    df_raw = load_telco_data()
    df_fe = engineer_features(df_raw)
    return df_raw, df_fe


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Cached loader for trained models and preprocessor."""
    artifacts = load_artifacts()
    if artifacts is None:
        # If not trained yet, run training pipeline
        df_raw, _ = get_telco_data()
        data_dict = prepare_churn_data(df_raw)
        train_and_save_artifacts(data_dict)
        artifacts = load_artifacts()
    return artifacts


@st.cache_resource(show_spinner=False)
def get_t_learner():
    """Cached loader for synthetic uplift T-Learner."""
    df_uplift = load_synthetic_uplift_data()
    return train_or_load_t_learner(df_uplift)


# Load core data & artifacts
df_raw, df_fe = get_telco_data()
artifacts = get_artifacts()
models_dict = artifacts["models"]
preprocessor = artifacts["preprocessor"]
metrics_df = artifacts["metrics_summary"]
test_eval_cache = artifacts["test_eval_cache"]

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand">ChurnIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Explainable Customer Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Customer Explorer",
            "3. Churn Prediction",
            "4. Explainability (XAI)",
            "5. What-if Analysis & DiCE",
            "6. Retention Decision Support",
            "7. Model Performance",
            "8. Research & Methodology"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### Active Model")
    selected_model_name = st.selectbox(
        "Scoring Model:",
        options=list(models_dict.keys()),
        index=list(models_dict.keys()).index("CatBoost") if "CatBoost" in models_dict else 0,
        help="Select model for real-time scoring and inference across the dashboard."
    )
    active_model = models_dict[selected_model_name]

    st.markdown("### Decision Threshold")
    risk_threshold = st.slider(
        "Churn Risk Threshold:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Custom probability threshold to classify a customer as churn-risk."
    )

    st.markdown("---")
    st.caption("B.Tech Technical Seminar Project")
    st.caption("PCET's PCCOE Pune • 2026")


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "1. Executive Overview":
    st.title("Executive Overview")
    st.markdown('<p class="page-subtitle">Management intelligence summary of customer cohort retention dynamics, risk distribution, and revenue concentration.</p>', unsafe_allow_html=True)

    # Score full dataset with active model for overview KPIs (cached)
    @st.cache_data(show_spinner=False)
    def get_scored_dataset(model_name: str):
        model = models_dict[model_name]
        return predict_batch(df_fe, model, preprocessor)

    df_scored = get_scored_dataset(selected_model_name)

    # Sidebar / Top Filters
    with st.expander("Filter Customer Cohort", expanded=False):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            contract_filter = st.multiselect(
                "Contract Type:",
                options=df_fe["Contract"].unique(),
                default=df_fe["Contract"].unique()
            )
        with col_f2:
            internet_filter = st.multiselect(
                "Internet Service:",
                options=df_fe["InternetService"].unique(),
                default=df_fe["InternetService"].unique()
            )
        with col_f3:
            payment_filter = st.multiselect(
                "Payment Method:",
                options=df_fe["PaymentMethod"].unique(),
                default=df_fe["PaymentMethod"].unique()
            )
        with col_f4:
            tenure_range = st.slider(
                "Tenure Range (Months):",
                min_value=int(df_fe["tenure"].min()),
                max_value=int(df_fe["tenure"].max()),
                value=(int(df_fe["tenure"].min()), int(df_fe["tenure"].max()))
            )

    # Apply filters
    filtered_df = df_scored[
        (df_scored["Contract"].isin(contract_filter)) &
        (df_scored["InternetService"].isin(internet_filter)) &
        (df_scored["PaymentMethod"].isin(payment_filter)) &
        (df_scored["tenure"].between(tenure_range[0], tenure_range[1]))
    ]

    # Distinguish actual vs predicted
    total_cust = len(filtered_df)
    actual_churn_count = (filtered_df["Churn"] == "Yes").sum()
    actual_churn_rate = (actual_churn_count / total_cust * 100) if total_cust > 0 else 0

    pred_churn_count = (filtered_df["Predicted_Churn_Prob"] >= risk_threshold).sum()
    pred_churn_rate = (pred_churn_count / total_cust * 100) if total_cust > 0 else 0
    high_risk_count = (filtered_df["Risk_Tier"] == "High").sum()
    revenue_at_risk = filtered_df[filtered_df["Predicted_Churn_Prob"] >= risk_threshold]["MonthlyCharges"].sum()

    # Top KPI Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card accent-neutral">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{total_cust:,}</div>
            <div class="kpi-subtext">Active cohort size</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card accent-danger">
            <div class="kpi-title">Actual Churn Rate</div>
            <div class="kpi-value">{actual_churn_rate:.1f}%</div>
            <div class="kpi-subtext">{actual_churn_count:,} observed churners</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card accent-warning">
            <div class="kpi-title">Predicted Churn Rate</div>
            <div class="kpi-value">{pred_churn_rate:.1f}%</div>
            <div class="kpi-subtext">At {risk_threshold:.0%} threshold ({selected_model_name})</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card accent-danger">
            <div class="kpi-title">High Risk Customers</div>
            <div class="kpi-value">{high_risk_count:,}</div>
            <div class="kpi-subtext">Risk score &gt;= 0.60</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col5:
        st.markdown(f"""
        <div class="kpi-card accent-warning">
            <div class="kpi-title">Monthly Revenue at Risk</div>
            <div class="kpi-value">${revenue_at_risk:,.0f}</div>
            <div class="kpi-subtext">Monthly recurring spend</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
        <b>Data Integrity Note:</b> Actual churn rate reflects ground-truth historical records, whereas predicted churn rate represents scoring from the active model pipeline. Statistics update dynamically when filters are applied.
    </div>
    """, unsafe_allow_html=True)

    # Interactive Visualizations
    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        st.plotly_chart(
            viz.plot_churn_distribution(filtered_df, col="Churn", title="Actual Churn Distribution (Ground Truth)"),
            use_container_width=True
        )
    with v_col2:
        # Predicted Churn Distribution
        pred_label_series = pd.Series(
            ["Predicted Churn" if p >= risk_threshold else "Predicted Retained" for p in filtered_df["Predicted_Churn_Prob"]]
        )
        counts_pred = pred_label_series.value_counts().reset_index()
        counts_pred.columns = ["Status", "Count"]
        colors_pred = ["#dc2626" if "Churn" in s else "#16a34a" for s in counts_pred["Status"]]
        fig_pred_pie = go.Figure(go.Pie(
            labels=counts_pred["Status"],
            values=counts_pred["Count"],
            hole=0.58,
            marker=dict(colors=colors_pred, line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#1e293b"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
        ))
        fig_pred_pie = viz._apply_defaults(fig_pred_pie,
            title=f"<b>Predicted Risk Distribution — {selected_model_name}</b>",
            margin=dict(t=44, b=12, l=12, r=12),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
            showlegend=True,
        )
        st.plotly_chart(fig_pred_pie, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(
            viz.plot_segment_churn(filtered_df, "Contract", title="Actual Churn Rate by Contract Commitment"),
            use_container_width=True
        )
    with row2_col2:
        st.plotly_chart(
            viz.plot_segment_churn(filtered_df, "InternetService", title="Actual Churn Rate by Internet Service Type"),
            use_container_width=True
        )

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        # Tenure Distribution Histogram
        fig_tenure = px.histogram(
            filtered_df,
            x="tenure",
            color="Churn",
            nbins=30,
            barmode="overlay",
            color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
        )
        fig_tenure = viz._apply_defaults(fig_tenure,
            title="<b>Tenure Distribution by Churn Status</b>",
            height=300,
            margin=dict(t=44, b=28, l=30, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    with row3_col2:
        # Monthly Charges Distribution
        fig_charges = px.histogram(
            filtered_df,
            x="MonthlyCharges",
            color="Churn",
            nbins=30,
            barmode="overlay",
            color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
        )
        fig_charges = viz._apply_defaults(fig_charges,
            title="<b>Monthly Charges Distribution by Churn Status</b>",
            height=300,
            margin=dict(t=44, b=28, l=30, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_charges, use_container_width=True)


# ==============================================================================
# PAGE 2: CUSTOMER EXPLORER
# ==============================================================================
elif page == "2. Customer Explorer":
    st.title("Customer Explorer")
    st.markdown('<p class="page-subtitle">Search, filter, and inspect individual customer accounts across observed and predicted dimensions.</p>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def get_explorer_dataset(model_name: str):
        model = models_dict[model_name]
        return predict_batch(df_fe, model, preprocessor)

    df_scored = get_explorer_dataset(selected_model_name)

    # Filter Controls
    search_col, filter_risk_col, filter_contract_col = st.columns([2, 1, 1])
    with search_col:
        search_query = st.text_input("Search by Customer ID:", placeholder="e.g. 7590-VHVEG or type to search...").strip()
    with filter_risk_col:
        risk_filter = st.selectbox("Filter Risk Tier:", ["All", "High", "Medium", "Low"])
    with filter_contract_col:
        contract_sub_filter = st.selectbox("Contract Type:", ["All"] + list(df_fe["Contract"].unique()))

    # Apply filters
    display_df = df_scored.copy()
    if search_query:
        display_df = display_df[display_df["customerID"].str.contains(search_query, case=False, na=False)]
    if risk_filter != "All":
        display_df = display_df[display_df["Risk_Tier"] == risk_filter]
    if contract_sub_filter != "All":
        display_df = display_df[display_df["Contract"] == contract_sub_filter]

    # Summary bar
    st.caption(f"Displaying **{len(display_df):,}** matching customers out of {len(df_scored):,} total records.")

    # Data Table View
    cols_to_show = [
        "customerID", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
        "InternetService", "PaymentMethod", "Churn", "Predicted_Churn_Prob", "Risk_Tier"
    ]
    formatted_table = display_df[cols_to_show].sort_values("Predicted_Churn_Prob", ascending=False)
    formatted_table["Predicted_Churn_Prob"] = formatted_table["Predicted_Churn_Prob"].apply(lambda p: f"{p:.2%}")
    formatted_table["MonthlyCharges"] = formatted_table["MonthlyCharges"].apply(lambda p: f"${p:.2f}")

    st.dataframe(
        formatted_table,
        use_container_width=True,
        height=320,
        column_config={
            "customerID": "Customer ID",
            "Churn": "Actual Churn",
            "Predicted_Churn_Prob": "Predicted Risk",
            "Risk_Tier": "Risk Tier"
        }
    )

    st.markdown("---")
    st.markdown("### Individual Customer Profile Inspector")

    selected_cid = st.selectbox(
        "Select Customer ID to Inspect Detailed Profile:",
        options=display_df["customerID"].tolist()[:100],
        help="Select a customer from the filtered list to open their complete diagnostic panel."
    )

    if selected_cid:
        cust_row = df_scored[df_scored["customerID"] == selected_cid].iloc[0]

        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        card_col1, card_col2, card_col3 = st.columns([1, 1, 1])
        with card_col1:
            st.markdown('<div class="profile-label">Customer ID</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["customerID"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Gender / Senior</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["gender"]} / {"Yes" if cust_row["SeniorCitizen"] == 1 else "No"}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Partner / Dependents</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["Partner"]} / {cust_row["Dependents"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Tenure</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["tenure"]} months ({cust_row["TenureGroup"]})</div>', unsafe_allow_html=True)
        with card_col2:
            st.markdown('<div class="profile-label">Contract</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["Contract"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Monthly Charges</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">${cust_row["MonthlyCharges"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Total Charges</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">${cust_row["TotalCharges"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Payment Method</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["PaymentMethod"]}</div>', unsafe_allow_html=True)
        with card_col3:
            actual_str = cust_row['Churn']
            actual_badge = "Churned" if actual_str == "Yes" else "Retained"
            prob = cust_row['Predicted_Churn_Prob']
            risk_tier = cust_row['Risk_Tier']
            badge_class = "badge-high" if risk_tier == "High" else ("badge-medium" if risk_tier == "Medium" else "badge-low")

            st.markdown('<div class="profile-label">Actual Status</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{actual_badge}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Predicted Churn Risk</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value" style="font-size:1.1rem;font-weight:700;color:#0f172a;">{prob:.2%}</div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Assigned Tier</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value"><span class="{badge_class}">{risk_tier} Risk</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="profile-label">Missing Support Services</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-value">{cust_row["SupportRiskCount"]} / 4</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Gauge Chart of Customer Risk
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Predicted Churn Probability", 'font': {'size': 14, 'color': '#334155'}},
            number={'suffix': "%", 'font': {'size': 28, 'color': '#0f172a'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 10, 'color': '#64748b'}},
                'bar': {'color': "#2563eb", 'thickness': 0.18},
                'steps': [
                    {'range': [0, 35], 'color': "#dcfce7"},
                    {'range': [35, 60], 'color': "#fef3c7"},
                    {'range': [60, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 3},
                    'thickness': 0.8,
                    'value': risk_threshold * 100
                }
            }
        ))
        fig_gauge = viz._apply_defaults(fig_gauge,
            height=240,
            margin=dict(t=30, b=10, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# PAGE 3: CHURN PREDICTION
# ==============================================================================
elif page == "3. Churn Prediction":
    st.title("Churn Prediction Studio")
    st.markdown('<p class="page-subtitle">Interactive inference engine supporting evaluation of existing accounts or real-time scoring of hypothetical prospective profiles.</p>', unsafe_allow_html=True)

    pred_mode = st.radio(
        "Select Scoring Mode:",
        ["A. Score Existing Dataset Customer", "B. Enter Custom / Prospective Customer Profile"],
        horizontal=True
    )

    customer_inputs = {}

    if pred_mode == "A. Score Existing Dataset Customer":
        st.markdown("#### Select Customer from Repository")
        sample_ids = df_raw["customerID"].tolist()
        selected_cust_id = st.selectbox("Search Customer ID:", options=sample_ids, index=0)

        cust_record = df_raw[df_raw["customerID"] == selected_cust_id].iloc[0]
        customer_inputs = cust_record.to_dict()

        with st.expander("View Extracted Customer Attributes", expanded=True):
            st.markdown('<div class="profile-card">', unsafe_allow_html=True)
            attr_c1, attr_c2, attr_c3 = st.columns(3)
            with attr_c1:
                st.markdown(f'<div class="profile-label">Contract</div><div class="profile-value">{customer_inputs.get("Contract")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-label">Tenure</div><div class="profile-value">{customer_inputs.get("tenure")} months</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-label">Monthly Charges</div><div class="profile-value">${customer_inputs.get("MonthlyCharges"):.2f}</div>', unsafe_allow_html=True)
            with attr_c2:
                st.markdown(f'<div class="profile-label">Internet Service</div><div class="profile-value">{customer_inputs.get("InternetService")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-label">Tech Support</div><div class="profile-value">{customer_inputs.get("TechSupport")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-label">Online Security</div><div class="profile-value">{customer_inputs.get("OnlineSecurity")}</div>', unsafe_allow_html=True)
            with attr_c3:
                st.markdown(f'<div class="profile-label">Payment Method</div><div class="profile-value">{customer_inputs.get("PaymentMethod")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-label">Paperless Billing</div><div class="profile-value">{customer_inputs.get("PaperlessBilling")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="profile-label">Actual Churn</div><div class="profile-value">{customer_inputs.get("Churn")}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("#### Enter Prospective Customer Attributes")
        with st.form("custom_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Demographics**")
                customer_inputs["gender"] = st.selectbox("Gender:", ["Female", "Male"])
                customer_inputs["SeniorCitizen"] = st.selectbox("Senior Citizen:", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                customer_inputs["Partner"] = st.selectbox("Has Partner:", ["Yes", "No"])
                customer_inputs["Dependents"] = st.selectbox("Has Dependents:", ["No", "Yes"])
                customer_inputs["tenure"] = st.slider("Tenure (Months):", min_value=0, max_value=72, value=12)

            with col2:
                st.markdown("**Core Telecom Services**")
                customer_inputs["PhoneService"] = st.selectbox("Phone Service:", ["Yes", "No"])
                customer_inputs["MultipleLines"] = st.selectbox("Multiple Lines:", ["No", "Yes", "No phone service"])
                customer_inputs["InternetService"] = st.selectbox("Internet Service:", ["Fiber optic", "DSL", "No"])
                customer_inputs["OnlineSecurity"] = st.selectbox("Online Security:", ["No", "Yes", "No internet service"])
                customer_inputs["OnlineBackup"] = st.selectbox("Online Backup:", ["No", "Yes", "No internet service"])
                customer_inputs["DeviceProtection"] = st.selectbox("Device Protection:", ["No", "Yes", "No internet service"])

            with col3:
                st.markdown("**Contract & Billing**")
                customer_inputs["TechSupport"] = st.selectbox("Tech Support:", ["No", "Yes", "No internet service"])
                customer_inputs["StreamingTV"] = st.selectbox("Streaming TV:", ["No", "Yes", "No internet service"])
                customer_inputs["StreamingMovies"] = st.selectbox("Streaming Movies:", ["No", "Yes", "No internet service"])
                customer_inputs["Contract"] = st.selectbox("Contract:", ["Month-to-month", "One year", "Two year"])
                customer_inputs["PaperlessBilling"] = st.selectbox("Paperless Billing:", ["Yes", "No"])
                customer_inputs["PaymentMethod"] = st.selectbox("Payment Method:", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                customer_inputs["MonthlyCharges"] = st.slider("Monthly Charges ($):", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
                # Compute default TotalCharges
                customer_inputs["TotalCharges"] = round(customer_inputs["MonthlyCharges"] * max(customer_inputs["tenure"], 1), 2)

            submitted = st.form_submit_button("Compute Churn Probability")

    # Run Prediction
    if customer_inputs:
        pred_res = predict_single_customer(
            customer_inputs,
            active_model,
            preprocessor,
            low_thresh=0.35,
            high_thresh=risk_threshold
        )

        st.markdown("### Prediction Diagnostic Output")
        res_c1, res_c2, res_c3 = st.columns([1, 1, 1])

        prob_val = pred_res["churn_probability"]

        with res_c1:
            st.markdown(f"""
            <div class="kpi-card accent-primary">
                <div class="kpi-title">Predicted Churn Risk</div>
                <div class="kpi-value">{prob_val:.2%}</div>
                <div class="kpi-subtext">Active Model: {selected_model_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_c2:
            pred_class = 1 if prob_val >= risk_threshold else 0
            class_str = "Predicted Churn (1)" if pred_class == 1 else "Predicted Retained (0)"
            st.markdown(f"""
            <div class="kpi-card accent-neutral">
                <div class="kpi-title">Classification Decision</div>
                <div class="kpi-value">{class_str}</div>
                <div class="kpi-subtext">Decision Threshold: {risk_threshold:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_c3:
            st.markdown(f"""
            <div class="kpi-card accent-{('danger' if prob_val >= risk_threshold else 'success')}">
                <div class="kpi-title">Assigned Risk Tier</div>
                <div class="kpi-value" style="color: {pred_res['risk_color']};">{pred_res['risk_badge']}</div>
                <div class="kpi-subtext">{pred_res['risk_summary']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-callout">
            <b>Suggested Next Step:</b> Proceed to the <b>Explainability (XAI)</b> tab to inspect the top factors influencing this probability, or jump to <b>What-if Analysis & DiCE</b> to simulate counterfactual retention actions.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: EXPLAINABILITY (XAI)
# ==============================================================================
elif page == "4. Explainability (XAI)":
    st.title("Explainable AI (XAI)")
    st.markdown('<p class="page-subtitle">Interpreting predictive mechanics through global model feature importance and granular individual attribution factors.</p>', unsafe_allow_html=True)

    tab_global, tab_local = st.tabs(["Global Feature Importance", "Individual Customer Attribution"])

    with tab_global:
        st.markdown("### Global Influence of Features across Population")
        st.markdown(
            "Feature importance reveals which variables the model relies upon most heavily when forming predictions across all customers."
        )

        feature_names = artifacts["feature_names"]
        df_imp = get_global_feature_importance(active_model, feature_names, top_n=15)

        g_col1, g_col2 = st.columns([3, 2])
        with g_col1:
            st.plotly_chart(
                viz.plot_feature_importance_bar(df_imp, model_name=selected_model_name),
                use_container_width=True
            )
        with g_col2:
            st.markdown("#### Plain-Language Feature Interpretation")
            st.markdown("""
            - **Contract Commitment**: Customers on Month-to-month contracts have negligible switching friction and account for the vast majority of churn incidents.
            - **Customer Tenure**: Early tenure (months 1-12) represents the highest vulnerability window. As tenure increases, organizational inertia dramatically lowers churn hazard.
            - **Monthly Charges**: High bills create recurring price awareness. In the presence of alternative promotions, high monthly spend accelerates defection.
            - **Support Ecosystem (TechSupport, OnlineSecurity)**: Customers who lack technical support or security add-ons exhibit significantly higher churn rates.
            """)
            st.markdown("""
            <div class="warning-callout">
                <b>Scientific Disclaimer:</b> Global feature importance reflects predictive statistical association within the trained model. It does <b>not</b> prove real-world causality.
            </div>
            """, unsafe_allow_html=True)

    with tab_local:
        st.markdown("### Local Attribution: Why is this customer predicted at risk?")
        st.markdown("Deconstructs an individual profile into risk drivers and protective anchors.")

        cid_to_explain = st.selectbox(
            "Choose Customer to Explain:",
            options=df_raw["customerID"].tolist()[:100],
            index=0
        )
        sample_cust = df_fe[df_fe["customerID"] == cid_to_explain].iloc[0]

        # Score customer
        cust_pred = predict_single_customer(sample_cust.to_dict(), active_model, preprocessor)
        explanation = explain_single_customer(sample_cust, df_fe)

        exp_c1, exp_c2 = st.columns([1, 1])
        with exp_c1:
            st.markdown(f"**Customer:** `{cid_to_explain}` &nbsp;|&nbsp; **Predicted Risk:** **{cust_pred['churn_probability']:.2%}** ({cust_pred['risk_badge']})")
            st.markdown("#### Primary Risk Factors")
            if explanation["risk_factors"]:
                for rf in explanation["risk_factors"]:
                    st.markdown(f'<div class="factor-card risk"><div class="factor-title">{rf["feature"]}</div><div class="factor-value">{rf["value"]}</div><div class="factor-desc">{rf["description"]}</div></div>', unsafe_allow_html=True)
            else:
                st.write("No major risk factors detected for this customer.")

        with exp_c2:
            st.markdown(f"**Actual Ground Truth:** `{'Churned' if sample_cust['Churn'] == 'Yes' else 'Retained'}`")
            st.markdown("#### Protective Factors")
            if explanation["protective_factors"]:
                for pf in explanation["protective_factors"]:
                    st.markdown(f'<div class="factor-card protective"><div class="factor-title">{pf["feature"]}</div><div class="factor-value">{pf["value"]}</div><div class="factor-desc">{pf["description"]}</div></div>', unsafe_allow_html=True)
            else:
                st.write("No significant protective anchors found.")


# ==============================================================================
# PAGE 5: WHAT-IF ANALYSIS & DiCE
# ==============================================================================
elif page == "5. What-if Analysis & DiCE":
    st.title("What-if Analysis & Counterfactuals")
    st.markdown('<p class="page-subtitle">Explore actionable counterfactual alternatives: what minimal changes to controllable features would flip the model prediction from Churned to Retained?</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
        <b>Methodological Guardrail:</b> A counterfactual explanation shows what hypothetical input changes would alter the <i>model's calculation</i>. It is an exploration of decision boundaries, not a guarantee that a real human will stay.
    </div>
    """, unsafe_allow_html=True)

    target_cid = st.selectbox(
        "Select High-Risk Customer for What-if Exploration:",
        options=df_raw["customerID"].tolist()[:50],
        index=0
    )
    cust_data = df_fe[df_fe["customerID"] == target_cid].iloc[0].to_dict()
    base_pred = predict_single_customer(cust_data, active_model, preprocessor)

    st.markdown(f'''
    <div class="info-callout">
        <b>Original Profile:</b> <code>{target_cid}</code> &nbsp;|&nbsp; Contract: <code>{cust_data.get('Contract')}</code> &nbsp;|&nbsp; Monthly: <code>${cust_data.get('MonthlyCharges'):.2f}</code> &nbsp;|&nbsp; Original Churn Probability: <b>{base_pred['churn_probability']:.2%}</b>
    </div>
    ''', unsafe_allow_html=True)

    tab_sim, tab_dice = st.tabs(["Feasible Business Retention Levers", "DiCE Optimization Engine"])

    with tab_sim:
        st.markdown("#### Realistic Business Scenarios Simulator")
        st.markdown(
            "Evaluates concrete commercial interventions (discounts, contract transitions, and service bundles) "
            "against the active machine learning model."
        )

        scenarios = simulate_business_what_if_scenarios(cust_data, active_model, preprocessor)

        for s in scenarios:
            st.markdown('<div class="scenario-row">', unsafe_allow_html=True)
            sc_col1, sc_col2, sc_col3, sc_col4 = st.columns([3, 1, 1, 2])
            with sc_col1:
                st.markdown(f"**{s['scenario']}**")
                for k, v in s["changes"].items():
                    st.caption(f"Change `{k}`: {v}")
            with sc_col2:
                st.markdown(f"**New Risk:** {s['new_prob']:.2%}")
                st.caption(f"Drop: **-{s['prob_reduction']:.2%}**")
            with sc_col3:
                st.markdown(f"**Tier:** {s['classification']}")
            with sc_col4:
                st.caption(f"Feasibility: {s['feasibility']}")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_dice:
        st.markdown("#### DiCE Counterfactual Exploration (Cell 7-11 Architecture)")
        st.markdown(
            "DiCE searches the feature space for diverse combinations of actionable features "
            "(`Contract`, `MonthlyCharges`, `PaymentMethod`, `PaperlessBilling`) that achieve retention class 0."
        )

        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            features_to_vary = st.multiselect(
                "Features Permitted to Vary:",
                options=ACTIONABLE_FEATURES,
                default=["Contract", "MonthlyCharges", "PaymentMethod", "PaperlessBilling"]
            )
        with col_d2:
            num_cfs = st.slider("Total Counterfactuals to Generate:", 1, 10, value=3)

        if st.button("Generate DiCE Counterfactuals", type="primary"):
            with st.spinner("Executing DiCE optimization search..."):
                try:
                    catboost_native = artifacts.get("catboost_native")
                    if catboost_native is None:
                        st.error("Native CatBoost model artifact not found.")
                    else:
                        dice_exp = init_dice_explainer(catboost_native, df_fe)
                        dice_result = generate_dice_counterfactuals(
                            dice_exp,
                            catboost_native,
                            pd.Series(cust_data),
                            features_to_vary=features_to_vary,
                            total_cfs=num_cfs
                        )

                        if dice_result["success"]:
                            st.success(f"Generated {len(dice_result['counterfactuals'])} diverse counterfactual candidates!")
                            for cf in dice_result["counterfactuals"]:
                                with st.expander(f"Counterfactual #{cf['cf_id']} — New Churn Probability: {cf['churn_prob']:.2%}", expanded=True):
                                    for feat, change in cf["changes"].items():
                                        st.markdown(f'''
                                        <div class="factor-card protective">
                                            <div class="factor-title">{feat}</div>
                                            <div class="factor-value">{change['from']} &rarr; {change['to']}</div>
                                        </div>
                                        ''', unsafe_allow_html=True)
                        else:
                            st.warning(f"DiCE Solver note: {dice_result.get('error', 'Unable to find valid CF in bounds')}")
                            st.info("Tip: You can also use the Feasible Business Retention Levers tab for immediate deterministic simulations.")
                except Exception as ex:
                    st.warning(f"DiCE computation note: {str(ex)}")
                    st.info("You can use the Business Levers tab above which guarantees instantaneous model scoring.")


# ==============================================================================
# PAGE 6: RETENTION DECISION SUPPORT
# ==============================================================================
elif page == "6. Retention Decision Support":
    st.title("Retention Decision Support")
    st.markdown('<p class="page-subtitle">Translating risk probability, feature drivers, and feasibility constraints into prioritized, cautious retention actions.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
        <b>Decision Philosophy:</b> Blindly offering heavy discounts to every predicted churner erodes profit margins. Intelligent decision support matches the intervention type to the underlying vulnerability.
    </div>
    """, unsafe_allow_html=True)

    rec_cid = st.selectbox(
        "Select Customer for Retention Strategy Formulation:",
        options=df_raw["customerID"].tolist()[:50],
        index=0
    )
    rec_cust = df_fe[df_fe["customerID"] == rec_cid].iloc[0].to_dict()
    rec_pred = predict_single_customer(rec_cust, active_model, preprocessor)

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        st.markdown(f"### Profile Summary")
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-label">Customer ID</div><div class="profile-value" style="font-size:1.05rem;font-weight:700;">{rec_cid}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-label">Contract</div><div class="profile-value">{rec_cust.get("Contract")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-label">Tenure</div><div class="profile-value">{rec_cust.get("tenure")} months</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-label">Monthly Bill</div><div class="profile-value">${rec_cust.get("MonthlyCharges"):.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-label">Predicted Churn Risk</div><div class="profile-value" style="font-size:1.15rem;font-weight:700;color:{rec_pred["risk_color"]};">{rec_pred["churn_probability"]:.2%} &mdash; {rec_pred["risk_badge"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown("### Recommended Action Plan")
        prob = rec_pred["churn_probability"]
        contract = rec_cust.get("Contract")
        support_risk = rec_cust.get("SupportRiskCount", 0)
        payment = str(rec_cust.get("PaymentMethod", ""))
        monthly = float(rec_cust.get("MonthlyCharges", 0))

        recommendations = []

        if prob >= 0.50 and contract == "Month-to-month":
            recommendations.append({
                "action": "Contract Migration Incentive",
                "priority": "High",
                "details": "Offer a 1-year commitment discount with 1st month free to bridge the vulnerable month-to-month period.",
                "type": "Rule-derived & Feature-targeted"
            })

        if support_risk >= 3:
            recommendations.append({
                "action": "TechSupport & Security Onboarding",
                "priority": "Medium",
                "details": "Customer lacks essential support services. Provide 6 months complimentary TechSupport & OnlineSecurity to increase stickiness.",
                "type": "Ecosystem Retention Lever"
            })

        if "Electronic check" in payment:
            recommendations.append({
                "action": "Autopay Migration Promotion",
                "priority": "Medium",
                "details": "Electronic check carries high friction. Offer a recurring $5 bill credit for setting up automated credit card or bank transfer billing.",
                "type": "Friction Reduction"
            })

        if prob < 0.35:
            recommendations.append({
                "action": "Standard Relationship Maintenance",
                "priority": "Low",
                "details": "Customer is well-anchored with low churn likelihood. Do not perturb with aggressive sales outreach.",
                "type": "Organic Loyalty"
            })

        for rec in recommendations:
            prio_class = f"rec-priority-{rec['priority'].lower()}"
            st.markdown('<div class="rec-card">', unsafe_allow_html=True)
            st.markdown(f"<span class='rec-priority-tag {prio_class}'>{rec['priority']} Priority</span>", unsafe_allow_html=True)
            st.markdown(f"**{rec['action']}**")
            st.write(rec["details"])
            st.caption(f"Strategy Type: {rec['type']}")
            st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: MODEL PERFORMANCE
# ==============================================================================
elif page == "7. Model Performance":
    st.title("Model Evaluation")
    st.markdown('<p class="page-subtitle">Rigorous comparative evaluation of the 5 candidate machine learning architectures on held-out test data (N=1,409).</p>', unsafe_allow_html=True)

    # Metrics Summary Table
    st.markdown("### Model Leaderboard (Test Set)")
    st.dataframe(
        metrics_df.round(4).sort_values("ROC-AUC", ascending=False),
        use_container_width=True
    )

    # Interactive Metric Comparison Bar
    metric_choice = st.selectbox(
        "Select Metric to Rank Models:",
        options=["ROC-AUC", "PR-AUC (AP)", "Accuracy", "F1 Score", "Precision", "Recall"],
        index=0
    )
    st.plotly_chart(viz.plot_model_comparison_bar(metrics_df, metric=metric_choice), use_container_width=True)

    # ROC and PR Curves
    c_roc, c_pr = st.columns(2)
    with c_roc:
        st.plotly_chart(viz.plot_multi_roc_curves(test_eval_cache), use_container_width=True)
    with c_pr:
        st.plotly_chart(viz.plot_multi_pr_curves(test_eval_cache), use_container_width=True)

    # Confusion Matrix for Selected Model
    st.markdown("### Confusion Matrix")
    eval_model_name = st.selectbox("Inspect Confusion Matrix for:", options=list(models_dict.keys()), index=0)
    cm_data = test_eval_cache["models"][eval_model_name]["cm"]
    st.plotly_chart(viz.plot_confusion_matrix_heatmap(cm_data, model_name=eval_model_name), use_container_width=True)

    # Analysis Commentary
    st.markdown("### Model Evaluation Analysis")
    st.markdown("""
    - **ROC-AUC Winner**: **Logistic Regression (0.8433)** achieved the highest ROC-AUC, closely followed by **CatBoost (0.8356)** and **LightGBM (0.8329)**.
    - **PR-AUC (Precision-Recall) Winner**: **CatBoost (0.6414)** achieved the best Average Precision, indicating superior calibration in ranking true churners in the presence of class imbalance (~26.5% churn).
    - **Selection Rationale for CatBoost**: While Logistic Regression achieved marginally higher ROC-AUC, CatBoost was selected for DiCE explainability and Uplift Modeling because of its native categorical handling, resistance to overfitting, and superior probability calibration.
    """)


# ==============================================================================
# PAGE 8: RESEARCH & METHODOLOGY
# ==============================================================================
elif page == "8. Research & Methodology":
    st.title("Research & Methodology")
    st.markdown('<p class="page-subtitle">Academic and technical seminar reference documentation detailing project formulation, causal uplift modeling, and experimental findings.</p>', unsafe_allow_html=True)

    st.markdown("### Project Title & Authorship")
    st.markdown("""
    **Explainable Customer Churn Prediction Using Machine Learning for Business Decision Support**  
    *Department of Computer Engineering, Pimpri Chinchwad College of Engineering (PCCOE), Pune*  
    *Authors:* Vivek Bhise, Aniket Gadekar, Piyush Jamdade, Harish Sanke • *Guide:* Dr. Minal Shahakar
    """)

    st.markdown("### End-to-End Analytics Pipeline")
    st.markdown("""
    <div class="pipeline-box">
    <pre style="margin:0;font-family:monospace;font-size:0.8rem;color:#334155;line-height:1.6;">
    ┌──────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
    │  Raw Customer Data   │ ───► │ Domain Feature Engineering│ ───► │ Churn Prediction Models │
    │ (Telco 7,043 records)│      │  (6 engineered features)  │      │(Logistic, RF, CatBoost) │
    └──────────────────────┘      └───────────────────────────┘      └────────────┬────────────┘
                                                                                      │
    ┌──────────────────────┐      ┌───────────────────────────┐      ┌────────────▼────────────┐
    │ Retention Decision   │ ◄─── │   What-if & Counterfactual│ ◄─── │ Global & Local XAI      │
    │   Support Strategy   │      │       (DiCE Framework)    │      │ (Feature Attribution)   │
    └──────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
    </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Causal Uplift Modeling (T-Learner Validation)")
    st.markdown("""
    <div class="warning-callout">
        <b>Crucial Research Insight:</b> Traditional churn models predict who is at risk of leaving, but <b>not</b> who can be saved by an offer. Uplift modeling estimates the Conditional Average Treatment Effect (CATE):
        <br><br>
        <code>τ(X) = P(Churn | X, No Offer) - P(Churn | X, With Offer)</code>
    </div>
    """, unsafe_allow_html=True)

    t_learner_data = get_t_learner()
    eval_res = t_learner_data["eval_results"]
    test_eval_df = eval_res["test_df"]

    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
    with u_col1:
        st.metric("Mean Estimated Uplift", f"{eval_res['mean_estimated_uplift']:.2%}")
    with u_col2:
        st.metric("Ground-Truth True Uplift", f"{eval_res['mean_true_uplift']:.2%}")
    with u_col3:
        st.metric("Spearman Rank Correlation", f"{eval_res['spearman_correlation']:.4f}")
    with u_col4:
        st.metric("Top 10% Targeting Gain", f"+{eval_res['uplift_improvement']:.2%}")

    u_chart1, u_chart2 = st.columns(2)
    with u_chart1:
        st.plotly_chart(viz.plot_causal_quadrant_scatter(test_eval_df), use_container_width=True)
    with u_chart2:
        st.plotly_chart(viz.plot_uplift_rank_deciles(test_eval_df), use_container_width=True)

    st.markdown("### Key Limitations & Future Scope")
    st.markdown("""
    1. **Synthetic Uplift Assumptions**: Uplift experimentation relies on synthetic potential outcomes (`synthetic_telco_uplift.csv`) since observational customer churn data lacks randomized retention trial ground truth.
    2. **Counterfactual Feasibility**: DiCE counterfactuals identify model boundaries, but business feasibility (e.g. willingness of customers to sign multi-year contracts) requires marketing validation.
    3. **Continuous Probability Calibration**: In production, probability calibration curves should be updated monthly to prevent concept drift.
    """)
