"""
models_manager.py - Model definitions, training, evaluation, serialization, and retrieval.
Faithfully mirrors the 5 models evaluated in TechnicalSem.ipynb.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def get_model_specifications():
    """Returns initialized model dictionary matching TechnicalSem.ipynb."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )
    }


def train_and_save_artifacts(data_dict: dict, models_dir: str = None):
    """
    Fits all 5 models on the processed dataset, computes exact evaluation curves,
    and serializes models and metrics for zero-latency dashboard operations.
    """
    save_dir = models_dir or MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    X_train_processed = data_dict["X_train_processed"]
    y_train = data_dict["y_train"]
    X_test_processed = data_dict["X_test_processed"]
    y_test = data_dict["y_test"]
    preprocessor = data_dict["preprocessor"]
    feature_names = data_dict["feature_names"]

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(save_dir, "feature_names.joblib"))

    models = get_model_specifications()
    results = []
    test_eval_cache = {
        "y_test": y_test.to_numpy(),
        "models": {}
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)

        # Save model artifact
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(save_dir, f"{safe_name}.joblib"), compress=3)

        # Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1 Score": float(f1),
            "ROC-AUC": float(roc),
            "PR-AUC (AP)": float(pr_auc)
        })

        test_eval_cache["models"][name] = {
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "prec_curve": prec_curve.tolist(),
            "rec_curve": rec_curve.tolist(),
            "cm": cm.tolist()
        }

    # Also train and save a CatBoost model directly on raw features for DiCE counterfactuals
    print("Training native CatBoost for DiCE counterfactuals...")
    from src.preprocessing import CATEGORICAL_FEATURES
    X_train_raw = data_dict["X_train"].copy()
    X_test_raw = data_dict["X_test"].copy()

    for col in CATEGORICAL_FEATURES:
        X_train_raw[col] = X_train_raw[col].fillna("Missing").astype(str)
        X_test_raw[col] = X_test_raw[col].fillna("Missing").astype(str)

    catboost_native = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    catboost_native.fit(X_train_raw, y_train, cat_features=CATEGORICAL_FEATURES)
    joblib.dump(catboost_native, os.path.join(save_dir, "catboost_native.joblib"), compress=3)

    # Save metrics summary
    metrics_df = pd.DataFrame(results).set_index("Model")
    joblib.dump(metrics_df, os.path.join(save_dir, "metrics_summary.joblib"))
    joblib.dump(test_eval_cache, os.path.join(save_dir, "test_eval_cache.joblib"))

    print("All models and artifacts successfully serialized.")
    return metrics_df


def load_artifacts(models_dir: str = None):
    """
    Loads saved models, preprocessor, and evaluation cache.
    If any artifact is missing, incompatible, or corrupted across environments,
    returns None to trigger clean automatic training.
    """
    save_dir = models_dir or MODELS_DIR
    required_files = [
        "metrics_summary.joblib",
        "preprocessor.joblib",
        "feature_names.joblib",
        "test_eval_cache.joblib",
        "logistic_regression.joblib",
        "random_forest.joblib",
        "lightgbm.joblib",
        "xgboost.joblib",
        "catboost.joblib"
    ]
    for rf in required_files:
        if not os.path.exists(os.path.join(save_dir, rf)):
            return None

    try:
        artifacts = {
            "preprocessor": joblib.load(os.path.join(save_dir, "preprocessor.joblib")),
            "feature_names": joblib.load(os.path.join(save_dir, "feature_names.joblib")),
            "metrics_summary": joblib.load(os.path.join(save_dir, "metrics_summary.joblib")),
            "test_eval_cache": joblib.load(os.path.join(save_dir, "test_eval_cache.joblib")),
            "models": {
                "Logistic Regression": joblib.load(os.path.join(save_dir, "logistic_regression.joblib")),
                "Random Forest": joblib.load(os.path.join(save_dir, "random_forest.joblib")),
                "LightGBM": joblib.load(os.path.join(save_dir, "lightgbm.joblib")),
                "XGBoost": joblib.load(os.path.join(save_dir, "xgboost.joblib")),
                "CatBoost": joblib.load(os.path.join(save_dir, "catboost.joblib")),
            }
        }
        if os.path.exists(os.path.join(save_dir, "catboost_native.joblib")):
            artifacts["catboost_native"] = joblib.load(os.path.join(save_dir, "catboost_native.joblib"))

        return artifacts
    except Exception as e:
        print(f"Notice: Artifact loading encountered environment incompatibility ({e}). Auto-training native models...")
        return None
