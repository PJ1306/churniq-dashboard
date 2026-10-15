"""
preprocessing.py - Feature engineering and preprocessing pipelines for ChurnIQ.
Faithfully implements the feature transformations from TechnicalSem.ipynb.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgChargesPerMonth",
    "ServiceCount",
    "ChargesPerTenure",
    "SupportRiskCount"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "ContractTenure"
]

ALL_FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the exact 6 domain feature engineering transformations
    from TechnicalSem.ipynb.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # 1. Tenure groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Short", "Medium", "Long"]
    ).astype("object")

    # 2. Average monthly spending
    df["AvgChargesPerMonth"] = (
        df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    )
    # If tenure is 0, fill with MonthlyCharges
    df["AvgChargesPerMonth"] = df["AvgChargesPerMonth"].fillna(df["MonthlyCharges"])

    # 3. Number of subscribed services
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    df["ServiceCount"] = sum(
        df[col].eq("Yes").astype(int)
        for col in service_cols
        if col in df.columns
    )

    # 4. Charges per tenure
    df["ChargesPerTenure"] = df["MonthlyCharges"] * df["tenure"]

    # 5. Contract and tenure interaction
    df["ContractTenure"] = (
        df["Contract"].fillna("Unknown").astype(str)
        + "_"
        + df["TenureGroup"].fillna("Unknown").astype(str)
    )

    # 6. Support-risk indicator
    risk_service_cols = [
        "OnlineSecurity",
        "TechSupport",
        "OnlineBackup",
        "DeviceProtection"
    ]
    available_risk_cols = [col for col in risk_service_cols if col in df.columns]
    df["SupportRiskCount"] = sum(
        df[col].eq("No").astype(int) for col in available_risk_cols
    )

    return df


def create_preprocessor():
    """
    Creates the Scikit-learn ColumnTransformer matching TechnicalSem.ipynb:
    - Numeric: SimpleImputer(median) -> StandardScaler()
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    return preprocessor


def prepare_churn_data(df: pd.DataFrame):
    """
    Full pipeline to engineer features and split into train/test sets.
    Returns:
        X_train, X_test, y_train, y_test, preprocessor, X_train_processed, X_test_processed, feature_names
    """
    df_fe = engineer_features(df)

    y = df_fe["Churn"].map({"Yes": 1, "No": 0})
    X = df_fe[ALL_FEATURE_COLS].copy()

    # Ensure categorical columns are strings
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Missing").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "feature_names": feature_names,
        "df_engineered": df_fe
    }
