# ChurnIQ — Explainable Customer Intelligence

> **Research Seminar Project**: Explainable Customer Churn Prediction Using Machine Learning for Business Decision Support  
> **Authors**: Vivek Bhise, Aniket Gadekar, Piyush Jamdade, Harish Sanke  
> **Guide**: Dr. Minal Shahakar  
> **Institution**: Department of Computer Engineering (Regional Language), PCET's Pimpri Chinchwad College of Engineering (PCCOE), Pune  

---

## 📌 Executive Summary

**ChurnIQ** transforms customer retention from a blunt risk-scoring mechanism into an explainable, decision-ready intelligence system. While traditional machine learning predicts **who** might leave, ChurnIQ integrates:
1. **Multi-Model Predictive Analytics**: 5 tuned classification architectures (Logistic Regression, CatBoost, LightGBM, Random Forest, XGBoost).
2. **Explainable AI (XAI)**: Global feature importance and individualized local factor attribution.
3. **What-if Counterfactual Analysis**: DiCE algorithmic counterfactuals and realistic commercial lever simulations.
4. **Causal Uplift Modeling (T-Learner)**: Estimating Conditional Average Treatment Effects (CATE) to target persuadable customers rather than wasting retention budget on lost causes.

---

## 🏗️ Repository Architecture

```
churn_research/
├── TechnicalSem.ipynb           # Original experimental Jupyter Notebook (Preserved as historical truth)
├── app.py                       # Complete 8-Page Streamlit Analytics Application
├── train_models.py              # Zero-latency serialization script for models and evaluation curves
├── requirements.txt             # Exact verified Python dependencies
├── README.md                    # Setup, execution, and academic methodology guide
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Benchmark Telco churn dataset (7,043 rows)
│   └── synthetic_telco_uplift.csv           # Causal uplift evaluation dataset with ground-truth potential outcomes
├── models/                      # Serialized models and evaluation artifacts
│   ├── logistic_regression.joblib
│   ├── catboost.joblib
│   ├── lightgbm.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── catboost_native.joblib   # Native CatBoost model for DiCE
│   ├── t_learner_control.joblib # T-Learner M0 model (No Offer)
│   ├── t_learner_treated.joblib # T-Learner M1 model (With Offer)
│   ├── preprocessor.joblib      # Scikit-learn ColumnTransformer pipeline
│   ├── metrics_summary.joblib   # Verified evaluation metrics table
│   └── test_eval_cache.joblib   # ROC, PR curves, and confusion matrices
└── src/
    ├── data_loader.py           # Data ingestion, type coercion, and validation
    ├── preprocessing.py         # 6 domain feature engineering transformations and pipeline
    ├── models_manager.py        # Model definitions, training, and artifact loading
    ├── prediction.py            # Single-customer scoring, batch inference, risk tiers
    ├── explainability.py        # Global importance & local attribution deconstruction
    ├── counterfactuals.py       # DiCE counterfactual generator & what-if simulator
    ├── uplift.py                # Causal T-Learner uplift modeling and decile analysis
    └── visualizations.py        # Interactive Plotly chart components
```

---

## ⚡ Quick Start Instructions (Windows)

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Navigate to the Project Directory
Open PowerShell or Command Prompt and run:
```powershell
cd C:\Users\pune\.gemini\antigravity\scratch\churn_research
```

### 3. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or activate using `cmd.exe /c .venv\Scripts\activate.bat`)*

### 4. (Optional) Re-train or Verify Artifacts
The repository already includes pre-trained model artifacts. If you wish to re-execute the training pipeline:
```powershell
python train_models.py
```

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 📊 Verified Model Evaluation Results

Evaluated on held-out test data ($N=1,409$, 20% stratified test split):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8020** | **0.6621** | **0.5588** | **0.6040** | **0.8433** | 0.6381 |
| **CatBoost** | 0.7892 | 0.6332 | 0.5241 | 0.5765 | 0.8356 | **0.6414** |
| **LightGBM** | 0.7935 | 0.6309 | 0.5428 | 0.5775 | 0.8329 | 0.6300 |
| **Random Forest** | 0.7800 | 0.6103 | 0.4733 | 0.5315 | 0.8253 | 0.6142 |
| **XGBoost** | 0.7708 | 0.5785 | 0.5187 | 0.5449 | 0.8094 | 0.5856 |

*Key Findings*:
- **Logistic Regression** achieved the highest overall ROC-AUC (0.8433).
- **CatBoost** achieved the highest Precision-Recall AUC (0.6414), demonstrating superior performance on ranking true churners under class imbalance.
- **CatBoost** was selected for DiCE and Uplift modeling due to its well-calibrated probability distributions and native categorical handling.

---

## 🔬 Uplift Modeling & Causal Decision Support

The T-Learner uplift framework estimates churn reduction under an intervention:
$$\hat{\tau}(X) = \hat{P}(\text{Churn} \mid X, T=0) - \hat{P}(\text{Churn} \mid X, T=1)$$

**Validation on Ground-Truth Synthetic Dataset**:
- **Mean Estimated Uplift**: $8.54\%$ (matches notebook: $0.08539$)
- **Mean Ground-Truth True Uplift**: $6.34\%$ (matches notebook: $0.06342$)
- **Spearman Rank Correlation**: **$0.5018$** (matches notebook: $0.50177$)
- **Top 10% Targeting Effectiveness**: **$12.53\%$ true churn reduction** vs **$6.34\%$ for random targeting** ($\mathbf{+6.19\%}$ gain).

---

## 🛡️ Limitations & Academic Guardrails

1. **Association vs. Causation**: Standard feature importance and churn probability indicate correlation in observational data. Only causal uplift and randomized A/B trials isolate intervention efficacy.
2. **Counterfactual Feasibility**: Changes suggested by DiCE explore model decision boundaries; business feasibility must be verified against organizational constraints.
3. **Synthetic Uplift Ground Truth**: Uplift modeling was validated against synthetic potential outcomes because real-world observational data cannot simultaneously record both treatment states for the same customer.
