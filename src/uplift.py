"""
uplift.py - Causal Inference & Uplift Modeling via T-Learner.
Faithfully integrates Cells 12-14 of TechnicalSem.ipynb.
Includes causal quadrant segmentation and synthetic validation benchmarks.
"""

import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def prepare_uplift_data(df: pd.DataFrame):
    """
    Extracts features, treatment, observed synthetic churn, and ground-truth validation metrics.
    """
    excluded = [
        "customerID",
        "Churn",
        "Treatment",
        "Baseline_Churn_Probability",
        "Potential_Churn_No_Offer",
        "Potential_Churn_With_Offer",
        "Observed_Synthetic_Churn",
        "True_Uplift",
        "Treatment_Propensity"
    ]

    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].copy()

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].fillna("Missing").astype(str)

    treatment = df["Treatment"].astype(int)
    y = df["Observed_Synthetic_Churn"].astype(int)

    return X, treatment, y, cat_cols


def train_or_load_t_learner(df_synthetic: pd.DataFrame, models_dir: str = None):
    """
    Trains or loads the Two-Model (T-Learner) architecture:
    - Model 0: P(Churn=1 | X, Treatment=0)
    - Model 1: P(Churn=1 | X, Treatment=1)
    """
    save_dir = models_dir or MODELS_DIR
    ctrl_path = os.path.join(save_dir, "t_learner_control.joblib")
    treat_path = os.path.join(save_dir, "t_learner_treated.joblib")
    eval_path = os.path.join(save_dir, "t_learner_eval.joblib")

    if os.path.exists(ctrl_path) and os.path.exists(treat_path) and os.path.exists(eval_path):
        try:
            return {
                "model_control": joblib.load(ctrl_path),
                "model_treated": joblib.load(treat_path),
                "eval_results": joblib.load(eval_path)
            }
        except Exception as e:
            print(f"Notice: T-Learner artifact loading encountered incompatibility ({e}). Re-fitting...")

    X, treatment, y, cat_cols = prepare_uplift_data(df_synthetic)

    X_train, X_test, t_train, t_test, y_train, y_test, idx_train, idx_test = (
        train_test_split(
            X, treatment, y, df_synthetic.index,
            test_size=0.20,
            random_state=42,
            stratify=treatment
        )
    )

    control_mask = (t_train == 0)
    treated_mask = (t_train == 1)

    print("Fitting T-Learner Control Model...")
    model_control = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_seed=42
    )
    model_control.fit(
        X_train.loc[control_mask],
        y_train.loc[control_mask],
        cat_features=cat_cols
    )

    print("Fitting T-Learner Treated Model...")
    model_treated = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_seed=42
    )
    model_treated.fit(
        X_train.loc[treated_mask],
        y_train.loc[treated_mask],
        cat_features=cat_cols
    )

    # Predictions
    p_no_offer = model_control.predict_proba(X_test)[:, 1]
    p_with_offer = model_treated.predict_proba(X_test)[:, 1]
    estimated_uplift = p_no_offer - p_with_offer

    # Validation against synthetic ground truth
    true_uplift = df_synthetic.loc[idx_test, "True_Uplift"].to_numpy()
    mean_est_uplift = float(np.mean(estimated_uplift))
    mean_true_uplift = float(np.mean(true_uplift))
    spearman_corr = float(spearmanr(estimated_uplift, true_uplift).statistic)

    eval_df = pd.DataFrame({
        "Index": idx_test,
        "Treatment": t_test.to_numpy(),
        "Actual_Synthetic_Churn": y_test.to_numpy(),
        "Predicted_Churn_No_Offer": p_no_offer,
        "Predicted_Churn_With_Offer": p_with_offer,
        "Estimated_Uplift": estimated_uplift,
        "True_Uplift": true_uplift,
        "Potential_Churn_No_Offer": df_synthetic.loc[idx_test, "Potential_Churn_No_Offer"].to_numpy(),
        "Potential_Churn_With_Offer": df_synthetic.loc[idx_test, "Potential_Churn_With_Offer"].to_numpy(),
    })

    # Top 10% targeting gain
    eval_df["Uplift_Rank"] = eval_df["Estimated_Uplift"].rank(ascending=False, method="first")
    n_top = max(1, int(np.ceil(0.10 * len(eval_df))))
    top_10 = eval_df.nsmallest(n_top, "Uplift_Rank")
    top_10_true = float(top_10["True_Uplift"].mean())
    benchmark_gain = top_10_true - mean_true_uplift

    # Assign Causal Quadrant
    def assign_quadrant(row):
        no_off = row["Predicted_Churn_No_Offer"]
        with_off = row["Predicted_Churn_With_Offer"]
        if row["Estimated_Uplift"] < -0.05:
            return "Sleeping Dog (Do Not Disturb)"
        elif no_off >= 0.5 and with_off < 0.5:
            return "Persuadable (Prime Target)"
        elif no_off >= 0.5 and with_off >= 0.5:
            return "Lost Cause (Avoid High Cost)"
        else:
            return "Sure Thing (Organic Retention)"

    eval_df["Causal_Segment"] = eval_df.apply(assign_quadrant, axis=1)

    eval_summary = {
        "mean_estimated_uplift": mean_est_uplift,
        "mean_true_uplift": mean_true_uplift,
        "spearman_correlation": spearman_corr,
        "top_10_mean_true_uplift": top_10_true,
        "random_benchmark": mean_true_uplift,
        "uplift_improvement": benchmark_gain,
        "test_df": eval_df
    }

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model_control, ctrl_path, compress=3)
    joblib.dump(model_treated, treat_path, compress=3)
    joblib.dump(eval_summary, eval_path, compress=3)

    return {
        "model_control": model_control,
        "model_treated": model_treated,
        "eval_results": eval_summary
    }
