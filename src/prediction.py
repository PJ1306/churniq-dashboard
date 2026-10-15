"""
prediction.py - Model inference, single-customer scoring, batch prediction, and risk categorization.
Ensures strict training-serving consistency.
"""

import pandas as pd
import numpy as np
from src.preprocessing import engineer_features, ALL_FEATURE_COLS, CATEGORICAL_FEATURES


def categorize_risk(prob: float, low_threshold: float = 0.35, high_threshold: float = 0.60) -> dict:
    """
    Categorizes predicted churn probability into transparent risk tiers.
    """
    if prob >= high_threshold:
        return {
            "tier": "High Risk",
            "color": "#dc2626",
            "badge": "High Risk",
            "summary": "Customer exhibits high churn vulnerability. Urgent retention action recommended."
        }
    elif prob >= low_threshold:
        return {
            "tier": "Medium Risk",
            "color": "#d97706",
            "badge": "Medium Risk",
            "summary": "Customer shows moderate churn indicators. Proactive engagement advised."
        }
    else:
        return {
            "tier": "Low Risk",
            "color": "#16a34a",
            "badge": "Low Risk",
            "summary": "Customer has low churn likelihood. Standard relationship maintenance sufficient."
        }


def predict_single_customer(customer_data: dict, model, preprocessor, low_thresh: float = 0.35, high_thresh: float = 0.60) -> dict:
    """
    Takes raw customer inputs (dict), applies feature engineering, transforms via preprocessor,
    and returns probability, class, risk tier, and explanation context.
    """
    df_raw = pd.DataFrame([customer_data])
    df_fe = engineer_features(df_raw)

    # Format categorical columns as strings
    for col in CATEGORICAL_FEATURES:
        if col in df_fe.columns:
            df_fe[col] = df_fe[col].fillna("Missing").astype(str)

    # Reorder columns strictly to match ALL_FEATURE_COLS
    X_single = df_fe[ALL_FEATURE_COLS].copy()

    X_proc = preprocessor.transform(X_single)
    prob = float(model.predict_proba(X_proc)[0, 1])
    decision_threshold = 0.50
    pred_class = int(prob >= decision_threshold)
    risk_info = categorize_risk(prob, low_thresh, high_thresh)

    return {
        "churn_probability": prob,
        "predicted_class": pred_class,
        "risk_tier": risk_info["tier"],
        "risk_badge": risk_info["badge"],
        "risk_color": risk_info["color"],
        "risk_summary": risk_info["summary"],
        "engineered_features": df_fe.iloc[0].to_dict()
    }


def predict_batch(df_engineered: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """
    Computes churn probability and predicted class for an entire dataset.
    Returns DataFrame with appended prediction columns.
    """
    df_copy = df_engineered.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna("Missing").astype(str)

    X = df_copy[ALL_FEATURE_COLS].copy()
    X_proc = preprocessor.transform(X)

    probs = model.predict_proba(X_proc)[:, 1]
    df_copy["Predicted_Churn_Prob"] = probs
    df_copy["Predicted_Churn_Class"] = (probs >= 0.50).astype(int)
    df_copy["Risk_Tier"] = [
        "High" if p >= 0.60 else ("Medium" if p >= 0.35 else "Low")
        for p in probs
    ]
    return df_copy
