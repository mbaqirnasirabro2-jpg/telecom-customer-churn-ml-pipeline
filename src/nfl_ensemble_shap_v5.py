import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

print("==========================================================")
print("   END-TO-END CUSTOMER CHURN PREDICTION & SHAP PIPELINE   ")
print("==========================================================\n")

# --- 1. REALISTIC TELECOM DATASET GENERATION ---
def generate_churn_dataset(n_samples: int = 3000) -> pd.DataFrame:
    np.random.seed(42)
    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.round(np.random.uniform(25.0, 120.0, n_samples), 2)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples, p=[0.45, 0.35, 0.20])
    tech_support = np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples
    )
    support_calls = np.random.poisson(lam=1.8, size=n_samples)

    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 30, n_samples), 2)
    total_charges = np.clip(total_charges, monthly_charges, None)

    # Realistic business churn probability formula
    logit = (
        -1.5
        + 1.3 * (contract == 'Month-to-month')
        - 0.8 * (contract == 'Two year')
        + 0.7 * (internet_service == 'Fiber optic')
        + 0.6 * (tech_support == 'No')
        + 0.35 * support_calls
        + 0.015 * monthly_charges
        - 0.035 * tenure
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        'customer_id': [f'CUST-{10000 + i}' for i in range(n_samples)],
        'tenure_months': tenure,
        'contract_type': contract,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'payment_method': payment_method,
        'support_calls_90d': support_calls,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })
    return df

df = generate_churn_dataset(3000)
df.to_csv("data/telecom_customer_churn.csv", index=False)
print(f"[INFO] Generated & saved dataset: {df.shape[0]} rows | Churn Rate: {df['churn'].mean()*100:.2f}%\n")

# --- 2. FEATURE ENGINEERING ---
df['avg_monthly_spend_ratio'] = df['total_charges'] / (df['tenure_months'] * df['monthly_charges'] + 1e-5)
df['charges_per_support_call'] = df['monthly_charges'] / (df['support_calls_90d'] + 1)
df['is_high_risk_contract'] = ((df['contract_type'] == 'Month-to-month') & (df['tenure_months'] < 12)).astype(int)
