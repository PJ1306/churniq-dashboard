"""
test_app.py - Automated verification and diagnostic test suite for ChurnIQ.
Ensures every module, model, pipeline, and visualization runs without error.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_loader import load_telco_data, load_synthetic_uplift_data
from src.preprocessing import engineer_features, prepare_churn_data, ALL_FEATURE_COLS
from src.models_manager import load_artifacts
from src.prediction import predict_single_customer, predict_batch, categorize_risk
from src.explainability import get_global_feature_importance, explain_single_customer
from src.counterfactuals import simulate_business_what_if_scenarios
from src.uplift import train_or_load_t_learner
import src.visualizations as viz


class TestChurnIQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_telco = load_telco_data()
        cls.df_fe = engineer_features(cls.df_telco)
        cls.artifacts = load_artifacts()
        if cls.artifacts is None:
            data_dict = prepare_churn_data(cls.df_telco)
            train_and_save_artifacts(data_dict)
            cls.artifacts = load_artifacts()
        cls.assertIsNotNone(cls.artifacts, "Artifacts should be successfully loaded from models/ directory.")
        cls.models = cls.artifacts["models"]
        cls.preprocessor = cls.artifacts["preprocessor"]

    def test_01_data_loading(self):
        self.assertEqual(len(self.df_telco), 7043)
        self.assertIn("TotalCharges", self.df_telco.columns)
        self.assertIn("Churn", self.df_telco.columns)

    def test_02_feature_engineering(self):
        for col in ["TenureGroup", "AvgChargesPerMonth", "ServiceCount", "ChargesPerTenure", "ContractTenure", "SupportRiskCount"]:
            self.assertIn(col, self.df_fe.columns)
        self.assertEqual(len(ALL_FEATURE_COLS), 25)

    def test_03_models_evaluation_metrics(self):
        metrics_df = self.artifacts["metrics_summary"]
        self.assertIn("CatBoost", metrics_df.index)
        self.assertIn("Logistic Regression", metrics_df.index)
        # Check verified ROC-AUC range
        cat_auc = metrics_df.loc["CatBoost", "ROC-AUC"]
        lr_auc = metrics_df.loc["Logistic Regression", "ROC-AUC"]
        self.assertGreater(cat_auc, 0.82)
        self.assertGreater(lr_auc, 0.83)

    def test_04_single_customer_prediction(self):
        sample_row = self.df_fe.iloc[0].to_dict()
        model = self.models["CatBoost"]
        res = predict_single_customer(sample_row, model, self.preprocessor)
        self.assertIn("churn_probability", res)
        self.assertTrue(0.0 <= res["churn_probability"] <= 1.0)
        self.assertIn("risk_tier", res)

    def test_05_custom_hypothetical_prediction(self):
        custom_cust = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 171.0
        }
        model = self.models["CatBoost"]
        res = predict_single_customer(custom_cust, model, self.preprocessor)
        self.assertGreater(res["churn_probability"], 0.50, "High-risk profile should score high churn probability.")

    def test_06_batch_prediction(self):
        model = self.models["CatBoost"]
        df_batch = predict_batch(self.df_fe.head(50), model, self.preprocessor)
        self.assertIn("Predicted_Churn_Prob", df_batch.columns)
        self.assertIn("Risk_Tier", df_batch.columns)
        self.assertEqual(len(df_batch), 50)

    def test_07_explainability(self):
        feature_names = self.artifacts["feature_names"]
        model = self.models["CatBoost"]
        df_imp = get_global_feature_importance(model, feature_names, top_n=10)
        self.assertEqual(len(df_imp), 10)
        self.assertIn("Normalized_Importance", df_imp.columns)

        # Local explanation
        sample_row = self.df_fe.iloc[0]
        local_exp = explain_single_customer(sample_row, self.df_fe)
        self.assertIn("risk_factors", local_exp)
        self.assertIn("protective_factors", local_exp)

    def test_08_what_if_simulation(self):
        sample_row = self.df_fe.iloc[0].to_dict()
        model = self.models["CatBoost"]
        scenarios = simulate_business_what_if_scenarios(sample_row, model, self.preprocessor)
        self.assertGreaterEqual(len(scenarios), 3)
        for s in scenarios:
            self.assertIn("new_prob", s)
            self.assertIn("prob_reduction", s)

    def test_09_uplift_modeling(self):
        df_uplift = load_synthetic_uplift_data()
        t_learner_data = train_or_load_t_learner(df_uplift)
        eval_res = t_learner_data["eval_results"]
        self.assertAlmostEqual(eval_res["mean_estimated_uplift"], 0.0854, delta=0.01)
        self.assertAlmostEqual(eval_res["spearman_correlation"], 0.5018, delta=0.01)
        self.assertAlmostEqual(eval_res["top_10_mean_true_uplift"], 0.1253, delta=0.01)

    def test_10_visualizations(self):
        fig1 = viz.plot_churn_distribution(self.df_fe)
        self.assertIsNotNone(fig1)

        fig2 = viz.plot_segment_churn(self.df_fe, "Contract")
        self.assertIsNotNone(fig2)

        metrics_df = self.artifacts["metrics_summary"]
        fig3 = viz.plot_model_comparison_bar(metrics_df, "ROC-AUC")
        self.assertIsNotNone(fig3)

        test_cache = self.artifacts["test_eval_cache"]
        fig4 = viz.plot_multi_roc_curves(test_cache)
        self.assertIsNotNone(fig4)


if __name__ == "__main__":
    unittest.main()
