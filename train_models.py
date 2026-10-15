"""
train_models.py - One-time training & serialization script.
Trains all 5 churn models and the 2-model T-Learner, validating against TechnicalSem.ipynb.
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_loader import load_telco_data, load_synthetic_uplift_data
from src.preprocessing import prepare_churn_data
from src.models_manager import train_and_save_artifacts
from src.uplift import train_or_load_t_learner


def main():
    print("=" * 60)
    print("ChurnIQ Model Training & Artifact Generation Pipeline")
    print("=" * 60)

    start_time = time.time()

    # 1. Load Data
    print("\n[Step 1/3] Loading Telco Customer Churn dataset...")
    df_telco = load_telco_data()
    print(f"Loaded {len(df_telco):,} rows, {len(df_telco.columns)} columns.")

    # 2. Preprocess & Feature Engineering
    print("\n[Step 2/3] Performing Feature Engineering & Split...")
    data_dict = prepare_churn_data(df_telco)
    print(f"Engineered feature shape: {data_dict['X_train'].shape[1]} features.")
    print(f"Encoded training shape: {data_dict['X_train_processed'].shape}")
    print(f"Encoded test shape: {data_dict['X_test_processed'].shape}")

    # Train all 5 classifiers
    print("\n[Step 3/3] Training and Serializing 5 Churn Models...")
    metrics_df = train_and_save_artifacts(data_dict)
    print("\nVerified Model Evaluation Results:")
    print(metrics_df.round(4).sort_values("ROC-AUC", ascending=False))

    # Train T-Learner for Uplift Modeling
    print("\n[Step 4/3] Training and Serializing T-Learner for Uplift Modeling...")
    df_uplift = load_synthetic_uplift_data()
    t_learner_data = train_or_load_t_learner(df_uplift)
    eval_res = t_learner_data["eval_results"]
    print(f"Mean Estimated Uplift: {eval_res['mean_estimated_uplift']:.4f}")
    print(f"Mean True Uplift: {eval_res['mean_true_uplift']:.4f}")
    print(f"Spearman Correlation: {eval_res['spearman_correlation']:.4f}")
    print(f"Top 10% Mean True Uplift: {eval_res['top_10_mean_true_uplift']:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining pipeline completed successfully in {elapsed:.1f} seconds!")


if __name__ == "__main__":
    main()
