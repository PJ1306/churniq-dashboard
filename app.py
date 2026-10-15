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

# Custom Professional Business Analytics CSS
CUSTOM_CSS = """
<style>
    /* Main container background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* KPI Card styling */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Status Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }

    /* Information callout */
    .info-callout {
        background-color: #f0fdf4;
        border-left: 4px solid #2a9d8f;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #1e293b;
    }
    .warning-callout {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #78350f;
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
    st.markdown("## 🌐 ChurnIQ")
    st.caption("Explainable Customer Intelligence Platform")
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
    st.markdown("### ⚙️ Active Model")
    selected_model_name = st.selectbox(
        "Scoring Model:",
        options=list(models_dict.keys()),
        index=list(models_dict.keys()).index("CatBoost") if "CatBoost" in models_dict else 0,
        help="Select model for real-time scoring and inference across the dashboard."
    )
    active_model = models_dict[selected_model_name]

    st.markdown("### 🎯 Decision Threshold")
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
    st.title("📈 Executive Overview")
    st.markdown(
        "Management intelligence summary of customer cohort retention dynamics, "
        "risk distribution, and revenue concentration."
    )

    # Score full dataset with active model for overview KPIs (cached)
    @st.cache_data(show_spinner=False)
    def get_scored_dataset(model_name: str):
        model = models_dict[model_name]
        return predict_batch(df_fe, model, preprocessor)

    df_scored = get_scored_dataset(selected_model_name)

    # Sidebar / Top Filters
    with st.expander("🔍 Filter Customer Cohort", expanded=False):
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
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{total_cust:,}</div>
            <div class="kpi-subtext">Active cohort size</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Actual Churn Rate</div>
            <div class="kpi-value">{actual_churn_rate:.1f}%</div>
            <div class="kpi-subtext">{actual_churn_count:,} observed churners</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Predicted Churn Rate</div>
            <div class="kpi-value">{pred_churn_rate:.1f}%</div>
            <div class="kpi-subtext">At {risk_threshold:.0%} threshold ({selected_model_name})</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High Risk Customers</div>
            <div class="kpi-value">{high_risk_count:,}</div>
            <div class="kpi-subtext">Risk score &gt;= 0.60</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monthly Rev. at Risk</div>
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
        colors_pred = ["#e63946" if "Churn" in s else "#2a9d8f" for s in counts_pred["Status"]]
        fig_pred_pie = go.Figure(go.Pie(
            labels=counts_pred["Status"],
            values=counts_pred["Count"],
            hole=0.55,
            marker=dict(colors=colors_pred),
            textinfo="label+percent"
        ))
        fig_pred_pie.update_layout(
            title=f"<b>Predicted Risk Distribution ({selected_model_name})</b>",
            margin=dict(t=40, b=20, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)"
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
            color_discrete_map={"No": "#2a9d8f", "Yes": "#e63946"},
            title="<b>Tenure Distribution by Churn Status</b>"
        )
        fig_tenure.update_layout(
            height=300,
            margin=dict(t=40, b=30, l=30, r=20),
            paper_bgcolor="rgba(0,0,0,0)"
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
            color_discrete_map={"No": "#2a9d8f", "Yes": "#e63946"},
            title="<b>Monthly Charges Distribution by Churn Status</b>"
        )
        fig_charges.update_layout(
            height=300,
            margin=dict(t=40, b=30, l=30, r=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_charges, use_container_width=True)


# ==============================================================================
# PAGE 2: CUSTOMER EXPLORER
# ==============================================================================
elif page == "2. Customer Explorer":
    st.title("👥 Customer Explorer")
    st.markdown("Search, filter, and inspect individual customer accounts across observed and predicted dimensions.")

    @st.cache_data(show_spinner=False)
    def get_explorer_dataset(model_name: str):
        model = models_dict[model_name]
        return predict_batch(df_fe, model, preprocessor)

    df_scored = get_explorer_dataset(selected_model_name)

    # Filter Controls
    search_col, filter_risk_col, filter_contract_col = st.columns([2, 1, 1])
    with search_col:
        search_query = st.text_input("🔍 Search by Customer ID:", placeholder="e.g. 7590-VHVEG or type to search...").strip()
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
    st.markdown("### 📋 Individual Customer Profile Inspector")

    selected_cid = st.selectbox(
        "Select Customer ID to Inspect Detailed Profile:",
        options=display_df["customerID"].tolist()[:100],
        help="Select a customer from the filtered list to open their complete diagnostic panel."
    )

    if selected_cid:
        cust_row = df_scored[df_scored["customerID"] == selected_cid].iloc[0]

        card_col1, card_col2, card_col3 = st.columns([1, 1, 1])
        with card_col1:
            st.markdown(f"**Customer ID:** `{cust_row['customerID']}`")
            st.markdown(f"**Gender / Senior:** {cust_row['gender']} / {'Yes' if cust_row['SeniorCitizen'] == 1 else 'No'}")
            st.markdown(f"**Partner / Dependents:** {cust_row['Partner']} / {cust_row['Dependents']}")
            st.markdown(f"**Tenure:** {cust_row['tenure']} months ({cust_row['TenureGroup']})")
        with card_col2:
            st.markdown(f"**Contract:** `{cust_row['Contract']}`")
            st.markdown(f"**Monthly Charges:** `${cust_row['MonthlyCharges']:.2f}`")
            st.markdown(f"**Total Charges:** `${cust_row['TotalCharges']:.2f}`")
            st.markdown(f"**Payment Method:** {cust_row['PaymentMethod']}")
        with card_col3:
            actual_str = cust_row['Churn']
            actual_badge = "🔴 Churned" if actual_str == "Yes" else "🟢 Retained"
            prob = cust_row['Predicted_Churn_Prob']
            risk_tier = cust_row['Risk_Tier']
            badge_class = "badge-high" if risk_tier == "High" else ("badge-medium" if risk_tier == "Medium" else "badge-low")

            st.markdown(f"**Actual Status:** {actual_badge}")
            st.markdown(f"**Predicted Churn Risk:** **{prob:.2%}**")
            st.markdown(f"**Assigned Tier:** <span class='{badge_class}'>{risk_tier} Risk</span>", unsafe_allow_html=True)
            st.markdown(f"**Missing Support Services:** {cust_row['SupportRiskCount']} / 4")

        # Gauge Chart of Customer Risk
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Predicted Churn Probability (%)", 'font': {'size': 15}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1d3557"},
                'steps': [
                    {'range': [0, 35], 'color': "#dcfce7"},
                    {'range': [35, 60], 'color': "#fef3c7"},
                    {'range': [60, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': risk_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(t=30, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# PAGE 3: CHURN PREDICTION
# ==============================================================================
elif page == "3. Churn Prediction":
    st.title("🎯 Churn Prediction Studio")
    st.markdown(
        "Interactive inference engine supporting evaluation of existing accounts or real-time scoring of hypothetical prospective profiles."
    )

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

        with st.expander("👁️ View Extracted Customer Attributes", expanded=True):
            attr_c1, attr_c2, attr_c3 = st.columns(3)
            with attr_c1:
                st.write(f"**Contract:** {customer_inputs.get('Contract')}")
                st.write(f"**Tenure:** {customer_inputs.get('tenure')} months")
                st.write(f"**Monthly Charges:** ${customer_inputs.get('MonthlyCharges'):.2f}")
            with attr_c2:
                st.write(f"**Internet Service:** {customer_inputs.get('InternetService')}")
                st.write(f"**Tech Support:** {customer_inputs.get('TechSupport')}")
                st.write(f"**Online Security:** {customer_inputs.get('OnlineSecurity')}")
            with attr_c3:
                st.write(f"**Payment Method:** {customer_inputs.get('PaymentMethod')}")
                st.write(f"**Paperless Billing:** {customer_inputs.get('PaperlessBilling')}")
                st.write(f"**Actual Churn:** {customer_inputs.get('Churn')}")

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

            submitted = st.form_submit_button("⚡ Compute Churn Probability")

    # Run Prediction
    if customer_inputs:
        pred_res = predict_single_customer(
            customer_inputs,
            active_model,
            preprocessor,
            low_thresh=0.35,
            high_thresh=risk_threshold
        )

        st.markdown("### 📊 Prediction Diagnostic Output")
        res_c1, res_c2, res_c3 = st.columns([1, 1, 1])

        prob_val = pred_res["churn_probability"]

        with res_c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Predicted Churn Risk</div>
                <div class="kpi-value">{prob_val:.2%}</div>
                <div class="kpi-subtext">Active Model: {selected_model_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_c2:
            pred_class = 1 if prob_val >= risk_threshold else 0
            class_str = "Predicted Churn (1)" if pred_class == 1 else "Predicted Retained (0)"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Classification Decision</div>
                <div class="kpi-value">{class_str}</div>
                <div class="kpi-subtext">Decision Threshold: {risk_threshold:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_c3:
            st.markdown(f"""
            <div class="kpi-card">
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
    st.title("🔬 Explainable AI (XAI) Architecture")
    st.markdown(
        "Interpreting predictive mechanics through global model feature importance "
        "and granular individual attribution factors."
    )

    tab_global, tab_local = st.tabs(["🌐 Global Feature Importance", "👤 Individual Customer Attribution"])

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
            st.markdown("#### 📖 Plain-Language Feature Interpretation")
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
            st.markdown(f"**Customer:** `{cid_to_explain}` | **Predicted Risk:** **{cust_pred['churn_probability']:.2%}** ({cust_pred['risk_badge']})")
            st.markdown("#### 🚨 Primary Risk Factors (Pushing toward Churn)")
            if explanation["risk_factors"]:
                for rf in explanation["risk_factors"]:
                    st.markdown(f"- **{rf['feature']}** (`{rf['value']}`): {rf['description']}")
            else:
                st.write("No major risk factors detected for this customer.")

        with exp_c2:
            st.markdown(f"**Actual Ground Truth:** `{'Churned' if sample_cust['Churn'] == 'Yes' else 'Retained'}`")
            st.markdown("#### 🛡️ Protective Factors (Anchoring Retention)")
            if explanation["protective_factors"]:
                for pf in explanation["protective_factors"]:
                    st.markdown(f"- **{pf['feature']}** (`{pf['value']}`): {pf['description']}")
            else:
                st.write("No significant protective anchors found.")


# ==============================================================================
# PAGE 5: WHAT-IF ANALYSIS & DiCE
# ==============================================================================
elif page == "5. What-if Analysis & DiCE":
    st.title("💡 What-if Analysis & DiCE Counterfactuals")
    st.markdown(
        "Explore actionable counterfactual alternatives: *What minimal changes to controllable features "
        "would flip the model's prediction from Churned to Retained?*"
    )

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

    st.markdown(
        f"**Original Profile:** `{target_cid}` | Contract: `{cust_data.get('Contract')}` | "
        f"Monthly: `${cust_data.get('MonthlyCharges'):.2f}` | "
        f"Original Churn Probability: **{base_pred['churn_probability']:.2%}**"
    )

    tab_sim, tab_dice = st.tabs(["🚀 Feasible Business Retention Levers", "🎲 DiCE Optimization Engine"])

    with tab_sim:
        st.markdown("#### Realistic Business Scenarios Simulator")
        st.markdown(
            "Evaluates concrete commercial interventions (discounts, contract transitions, and service bundles) "
            "against the active machine learning model."
        )

        scenarios = simulate_business_what_if_scenarios(cust_data, active_model, preprocessor)

        for s in scenarios:
            with st.container():
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
                st.markdown("<hr style='margin: 8px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

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

        if st.button("⚡ Generate DiCE Counterfactuals", type="primary"):
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
                                    st.write("**Modified Features:**")
                                    for feat, change in cf["changes"].items():
                                        st.write(f"- `{feat}`: `{change['from']}` ➔ **`{change['to']}`**")
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
    st.title("🛡️ Retention Decision Support System")
    st.markdown(
        "Translating risk probability, feature drivers, and feasibility constraints into prioritized, cautious retention actions."
    )

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
        st.markdown(f"### Profile Summary: `{rec_cid}`")
        st.write(f"- **Contract:** {rec_cust.get('Contract')}")
        st.write(f"- **Tenure:** {rec_cust.get('tenure')} months")
        st.write(f"- **Monthly Bill:** ${rec_cust.get('MonthlyCharges'):.2f}")
        st.write(f"- **Predicted Churn Risk:** **{rec_pred['churn_probability']:.2%}** ({rec_pred['risk_badge']})")

    with col_r2:
        st.markdown("### 📋 Recommended Action Plan")
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
            st.markdown(f"**[{rec['priority']} Priority] {rec['action']}**")
            st.write(rec["details"])
            st.caption(f"Strategy Type: {rec['type']}")
            st.markdown("<hr style='margin: 8px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: MODEL PERFORMANCE
# ==============================================================================
elif page == "7. Model Performance":
    st.title("📊 Research Model Evaluation Dashboard")
    st.markdown(
        "Rigorous comparative evaluation of the 5 candidate machine learning architectures "
        "implemented in `TechnicalSem.ipynb` on held-out test data (N=1,409)."
    )

    # Metrics Summary Table
    st.markdown("### 🏆 Model Leaderboard (Test Set)")
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
    st.markdown("### 🔲 Confusion Matrix")
    eval_model_name = st.selectbox("Inspect Confusion Matrix for:", options=list(models_dict.keys()), index=0)
    cm_data = test_eval_cache["models"][eval_model_name]["cm"]
    st.plotly_chart(viz.plot_confusion_matrix_heatmap(cm_data, model_name=eval_model_name), use_container_width=True)

    # Analysis Commentary
    st.markdown("### 🔍 Model Evaluation Analysis")
    st.markdown("""
    - **ROC-AUC Winner**: **Logistic Regression (0.8433)** achieved the highest ROC-AUC, closely followed by **CatBoost (0.8356)** and **LightGBM (0.8329)**.
    - **PR-AUC (Precision-Recall) Winner**: **CatBoost (0.6414)** achieved the best Average Precision, indicating superior calibration in ranking true churners in the presence of class imbalance (~26.5% churn).
    - **Selection Rationale for CatBoost**: While Logistic Regression achieved marginally higher ROC-AUC, CatBoost was selected for DiCE explainability and Uplift Modeling because of its native categorical handling, resistance to overfitting, and superior probability calibration.
    """)


# ==============================================================================
# PAGE 8: RESEARCH & METHODOLOGY
# ==============================================================================
elif page == "8. Research & Methodology":
    st.title("🎓 Research & Methodology")
    st.markdown(
        "Academic and technical seminar reference documentation detailing project formulation, "
        "causal uplift modeling, and experimental findings."
    )

    st.markdown("### 📌 Project Title & Authorship")
    st.markdown("""
    **Explainable Customer Churn Prediction Using Machine Learning for Business Decision Support**  
    *Department of Computer Engineering, Pimpri Chinchwad College of Engineering (PCCOE), Pune*  
    *Authors:* Vivek Bhise, Aniket Gadekar, Piyush Jamdade, Harish Sanke • *Guide:* Dr. Minal Shahakar
    """)

    st.markdown("### 🔄 Complete End-to-End Analytics Pipeline")
    st.markdown("""
    ```
    ┌──────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
    │  Raw Customer Data   │ ───► │ Domain Feature Engineering│ ───► │ Churn Prediction Models │
    │ (Telco 7,043 records)│      │  (6 engineered features)  │      │(Logistic, RF, CatBoost) │
    └──────────────────────┘      └───────────────────────────┘      └────────────┬────────────┘
                                                                                  │
    ┌──────────────────────┐      ┌───────────────────────────┐      ┌────────────▼────────────┐
    │ Retention Decision   │ ◄─── │   What-if & Counterfactual│ ◄─── │ Global & Local XAI      │
    │   Support Strategy   │      │       (DiCE Framework)    │      │ (Feature Attribution)   │
    └──────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
    ```
    """)

    st.markdown("### 🔬 Causal Uplift Modeling (T-Learner Validation)")
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

    st.markdown("### ⚠️ Key Limitations & Future Scope")
    st.markdown("""
    1. **Synthetic Uplift Assumptions**: Uplift experimentation relies on synthetic potential outcomes (`synthetic_telco_uplift.csv`) since observational customer churn data lacks randomized retention trial ground truth.
    2. **Counterfactual Feasibility**: DiCE counterfactuals identify model boundaries, but business feasibility (e.g. willingness of customers to sign multi-year contracts) requires marketing validation.
    3. **Continuous Probability Calibration**: In production, probability calibration curves should be updated monthly to prevent concept drift.
    """)
