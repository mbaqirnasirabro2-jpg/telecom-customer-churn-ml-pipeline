# End-to-End Telecom Customer Churn Prediction & Retention Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-orange?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-red)

## 📌 Project Overview
Customer churn directly impacts recurring revenue in subscription-based telecom businesses. This end-to-end machine learning project builds a complete predictive analytics workflow—from synthetic data generation and exploratory data analysis (EDA) to multi-model benchmarking (**Logistic Regression**, **Random Forest**, **LightGBM**), **SHAP model explainability**, and automated generation of a **high-risk customer retention list**.

---

## 🏗️ Workflow Architecture

```text
+-----------------------+      +-----------------------+      +---------------------------+
|  Telecom Customer     | ---> |  Feature Engineering  | ---> |  Scikit-Learn Pipeline    |
|  Dataset (3,000 rows) |      |  & EDA Visualizations |      |  (Scaler + OneHotEncoder) |
+-----------------------+      +-----------------------+      +-------------+-------------+
                                                                            |
                                                                            v
+-----------------------+      +-----------------------+      +---------------------------+
|  High-Risk Retention  | <--- |  SHAP Explainability  | <--- |  Model Benchmarking       |
|  Target CSV Export    |      |  (Key Churn Drivers)  |      |  (LR vs RF vs LightGBM)   |
+-----------------------+      +-----------------------+      +---------------------------+
