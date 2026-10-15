"""
data_loader.py - Data ingestion and validation for ChurnIQ.
Loads the Telco Customer Churn dataset and Synthetic Uplift dataset.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TELCO_CSV = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
SYNTHETIC_UPLIFT_CSV = os.path.join(DATA_DIR, "synthetic_telco_uplift.csv")


def load_telco_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads and performs initial type conversion on the Telco Churn dataset.
    Preserves customerID and raw columns while coercing numeric fields.
    """
    path = filepath or TELCO_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Telco churn dataset not found at {path}")

    df = pd.read_csv(path)

    # Convert TotalCharges before feature engineering, coerce whitespace/errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Fill SeniorCitizen as integer
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    return df


def load_synthetic_uplift_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads the synthetic uplift dataset containing treatment and ground-truth counterfactuals.
    """
    path = filepath or SYNTHETIC_UPLIFT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Synthetic uplift dataset not found at {path}")

    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df
