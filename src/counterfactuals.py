"""
counterfactuals.py - DiCE counterfactual explanations and actionable what-if scenario simulations.
Faithfully integrates Cell 7-11 of TechnicalSem.ipynb with robust constraints.
"""

import pandas as pd
import numpy as np
import dice_ml
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURE_COLS, engineer_features


# Realistically controllable business levers
ACTIONABLE_FEATURES = [
    "Contract",
    "MonthlyCharges",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity",
    "OnlineBackup"
]

NON_ACTIONABLE_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "TotalCharges",
    "PhoneService",
    "MultipleLines",
    "InternetService"
]


def init_dice_explainer(catboost_native, train_df: pd.DataFrame):
    """
    Initializes DiCE using the native CatBoost model and training DataFrame
    matching Cell 7 of TechnicalSem.ipynb.
    """
    X_train_cf = train_df[ALL_FEATURE_COLS].copy()
    dice_train = X_train_cf.copy()
    dice_train["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0}).to_numpy()

    continuous_features = [
        col for col in NUMERIC_FEATURES if col in X_train_cf.columns
    ]

    for col in CATEGORICAL_FEATURES:
        if col in dice_train.columns:
            dice_train[col] = dice_train[col].fillna("Missing").astype(str)
            X_train_cf[col] = X_train_cf[col].fillna("Missing").astype(str)

    dice_data = dice_ml.Data(
        dataframe=dice_train,
        continuous_features=continuous_features,
        outcome_name="Churn"
    )

    dice_model = dice_ml.Model(
        model=catboost_native,
        backend="sklearn",
        model_type="classifier"
    )

    dice_exp = dice_ml.Dice(
        dice_data,
        dice_model,
        method="random"
    )
    return dice_exp


def generate_dice_counterfactuals(
    dice_exp,
    catboost_native,
    customer_series: pd.Series,
    features_to_vary: list = None,
    total_cfs: int = 5
) -> dict:
    """
    Generates actionable counterfactual instances using DiCE.
    Validates model output probability on each alternative.
    """
    vary_cols = features_to_vary or ["Contract", "MonthlyCharges", "PaymentMethod", "PaperlessBilling"]
    vary_cols = [c for c in vary_cols if c in ALL_FEATURE_COLS]

    customer_df = pd.DataFrame([customer_series])
    # Ensure correct columns and types
    customer_input = customer_df[ALL_FEATURE_COLS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in customer_input.columns:
            customer_input[col] = customer_input[col].fillna("Missing").astype(str)

    try:
        cf_exp = dice_exp.generate_counterfactuals(
            customer_input,
            total_CFs=total_cfs,
            features_to_vary=vary_cols,
            desired_class=0,
            posthoc_sparsity_param=0.1
        )

        cfs_df = cf_exp.cf_examples_list[0].final_cfs_df
        if cfs_df is None or cfs_df.empty:
            return {"success": False, "error": "No counterfactuals found by DiCE within constraints."}

        # Clean outcome column if present
        clean_cfs = cfs_df.drop(columns=["Churn", "outcome"], errors="ignore")
        clean_cfs = clean_cfs[ALL_FEATURE_COLS]

        # Calculate exact probabilities on counterfactual instances
        cf_probs = catboost_native.predict_proba(clean_cfs)[:, 1]

        # Compare diffs
        comparisons = []
        for i, row in clean_cfs.iterrows():
            changes = {}
            for col in vary_cols:
                orig_val = customer_input[col].iloc[0]
                cf_val = row[col]
                if str(orig_val) != str(cf_val):
                    changes[col] = {"from": orig_val, "to": cf_val}

            comparisons.append({
                "cf_id": i + 1,
                "changes": changes,
                "churn_prob": float(cf_probs[i]),
                "predicted_class": int(cf_probs[i] >= 0.50),
                "full_record": row.to_dict()
            })

        return {
            "success": True,
            "counterfactuals": comparisons,
            "raw_cfs_df": clean_cfs,
            "method": "DiCE (Random Search)"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def simulate_business_what_if_scenarios(
    customer_dict: dict,
    model,
    preprocessor
) -> list:
    """
    Simulates concrete, feasible business retention scenarios:
    1. Contract Extension (Switch to 1-Year or 2-Year Contract)
    2. Pricing Concession (10% - 20% discount on MonthlyCharges)
    3. Service Ecosystem Bundle (Add TechSupport & OnlineSecurity)
    4. Auto-Pay Onboarding (Switch from Electronic Check to Bank Transfer / Credit Card)
    5. Combined Retention Package (1-Year Contract + 10% Discount + TechSupport)
    """
    from src.prediction import predict_single_customer

    base_prediction = predict_single_customer(customer_dict, model, preprocessor)
    base_prob = base_prediction["churn_probability"]

    scenarios = []

    # Scenario 1: Upgrade Contract to One Year
    if customer_dict.get("Contract") != "Two year":
        s1 = customer_dict.copy()
        target_contract = "One year" if s1.get("Contract") == "Month-to-month" else "Two year"
        s1["Contract"] = target_contract
        pred = predict_single_customer(s1, model, preprocessor)
        scenarios.append({
            "scenario": f"Commitment Incentive: Upgrade to {target_contract}",
            "changes": {"Contract": f"{customer_dict.get('Contract')} → {target_contract}"},
            "new_prob": pred["churn_probability"],
            "prob_reduction": base_prob - pred["churn_probability"],
            "feasibility": "High (Offer 1st month free or device perk for commitment)",
            "classification": pred["risk_tier"]
        })

    # Scenario 2: 15% Pricing Concession
    monthly = float(customer_dict.get("MonthlyCharges", 70.0))
    discounted_monthly = round(monthly * 0.85, 2)
    s2 = customer_dict.copy()
    s2["MonthlyCharges"] = discounted_monthly
    pred2 = predict_single_customer(s2, model, preprocessor)
    scenarios.append({
        "scenario": "Price Adjustment: 15% Monthly Loyalty Discount",
        "changes": {"MonthlyCharges": f"${monthly:.2f} → ${discounted_monthly:.2f}"},
        "new_prob": pred2["churn_probability"],
        "prob_reduction": base_prob - pred2["churn_probability"],
        "feasibility": "Medium (Requires business margin approval)",
        "classification": pred2["risk_tier"]
    })

    # Scenario 3: Add TechSupport & OnlineSecurity
    s3 = customer_dict.copy()
    s3["TechSupport"] = "Yes"
    s3["OnlineSecurity"] = "Yes"
    pred3 = predict_single_customer(s3, model, preprocessor)
    scenarios.append({
        "scenario": "Product Sticky Bundle: Add Tech Support & Security",
        "changes": {
            "TechSupport": f"{customer_dict.get('TechSupport')} → Yes",
            "OnlineSecurity": f"{customer_dict.get('OnlineSecurity')} → Yes"
        },
        "new_prob": pred3["churn_probability"],
        "prob_reduction": base_prob - pred3["churn_probability"],
        "feasibility": "High (Low marginal cost for SaaS/telecom support)",
        "classification": pred3["risk_tier"]
    })

    # Scenario 4: Migrate Payment Method to Autopay
    if "automatic" not in str(customer_dict.get("PaymentMethod", "")).lower():
        s4 = customer_dict.copy()
        s4["PaymentMethod"] = "Bank transfer (automatic)"
        pred4 = predict_single_customer(s4, model, preprocessor)
        scenarios.append({
            "scenario": "Billing Friction Elimination: Switch to Auto-Pay",
            "changes": {"PaymentMethod": f"{customer_dict.get('PaymentMethod')} → Bank transfer (automatic)"},
            "new_prob": pred4["churn_probability"],
            "prob_reduction": base_prob - pred4["churn_probability"],
            "feasibility": "High (Offer $5 recurring autopay credit)",
            "classification": pred4["risk_tier"]
        })

    # Scenario 5: Comprehensive Retention Offer (Contract + Discount + Support)
    s5 = customer_dict.copy()
    s5["Contract"] = "One year"
    s5["MonthlyCharges"] = discounted_monthly
    s5["TechSupport"] = "Yes"
    pred5 = predict_single_customer(s5, model, preprocessor)
    scenarios.append({
        "scenario": "Premier Retention Package (1-Yr Term + 15% Off + Tech Support)",
        "changes": {
            "Contract": f"{customer_dict.get('Contract')} → One year",
            "MonthlyCharges": f"${monthly:.2f} → ${discounted_monthly:.2f}",
            "TechSupport": f"{customer_dict.get('TechSupport')} → Yes"
        },
        "new_prob": pred5["churn_probability"],
        "prob_reduction": base_prob - pred5["churn_probability"],
        "feasibility": "High Priority Retention Action for High-LTV accounts",
        "classification": pred5["risk_tier"]
    })

    return scenarios
