# 💰 Credit Risk Assessment using Predictive Analytics

> **AI-Driven Credit Risk Intelligence System** | Machine Learning | Python | Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34-red)](https://streamlit.io)

---

## 🎯 Problem Statement

Banks and financial institutions face enormous risk when lending money. A poor credit decision can lead to significant financial loss. This project builds a **machine learning system** that predicts whether a loan applicant will **default** on their loan, enabling lenders to make data-driven decisions.

---

## 📁 Project Structure

```
Credit-Risk-Analytics-Platform/
├── data/
│   └── credit_data.csv          ← Auto-generated realistic dataset (5,000 rows)
├── src/
│   ├── data_loader.py           ← Dataset generation & loading
│   ├── preprocessing.py         ← Imputation, encoding, scaling, SMOTE
│   ├── eda.py                   ← Exploratory Data Analysis (9 plots)
│   ├── train.py                 ← Model training (4 algorithms)
│   ├── evaluate.py              ← Metrics, ROC curve, confusion matrix
│   └── pipeline.py              ← ✅ Master pipeline (run this first!)
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl           ← Best model by ROC-AUC
│   ├── scaler.pkl
│   ├── imputer.pkl
│   └── feature_columns.pkl
├── reports/
│   ├── 01_class_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_xgboost.png
│   └── metrics_summary.csv
├── app/
│   └── app.py                   ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (3 Commands)

```bash
# Step 1 – Install dependencies
pip install -r requirements.txt

# Step 2 – Train all models (generates data, runs EDA, trains, evaluates)
python src/pipeline.py

# Step 3 – Launch the interactive web app
streamlit run app/app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## 📊 Dataset

**Source:** Synthetic dataset based on the **German Credit Dataset** + **Lending Club** structure.

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Applicant age (20–70) |
| `income` | Numeric | Annual income in USD |
| `loan_amount` | Numeric | Requested loan amount |
| `loan_term` | Numeric | Repayment period (months) |
| `credit_score` | Numeric | FICO credit score (300–850) |
| `employment_years` | Numeric | Years at current employer |
| `num_credit_lines` | Numeric | Total open credit accounts |
| `debt_to_income` | Numeric | Monthly debt / monthly income |
| `num_delinquencies` | Numeric | Number of past delinquent payments |
| `previous_defaults` | Numeric | Number of prior loan defaults |
| `education` | Categorical | Highest education level |
| `home_ownership` | Categorical | Rent / Own / Mortgage |
| `loan_purpose` | Categorical | Purpose of the loan |
| `employment_status` | Categorical | Full-time / Part-time / Self-employed / Unemployed |
| **`default`** | **Target** | **1 = Default (High Risk), 0 = No Default (Low Risk)** |

**Engineered Features:**
- `loan_to_income` = loan_amount / income
- `monthly_payment` = Amortization formula
- `payment_to_income` = monthly_payment / (income/12)

---

## 🤖 Machine Learning Models

| Model | Description | Strength |
|-------|-------------|----------|
| Logistic Regression | Linear probabilistic classifier | Fast, interpretable |
| Decision Tree | Rule-based tree | Explainable decisions |
| Random Forest | Ensemble of 300 trees | Robust, handles noise |
| XGBoost | Gradient boosting | Highest accuracy |

**Techniques applied:**
- ✅ SMOTE (Synthetic Minority Oversampling) for class imbalance
- ✅ `class_weight='balanced'` on Logistic Regression & Decision Tree
- ✅ StandardScaler for numeric feature normalization
- ✅ Median imputation for missing values
- ✅ One-hot encoding for categorical variables
- ✅ 5-fold cross-validation during training

---

## 📈 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/Total | Overall correctness |
| Precision | TP/(TP+FP) | Avoid false alarms |
| Recall | TP/(TP+FN) | Catch all defaults |
| F1-Score | 2×(P×R)/(P+R) | Balance P & R |
| ROC-AUC | Area under ROC | Overall discrimination power |

> **Business Note:** In credit risk, **Recall is most critical** — missing a default is more costly than a false alarm.

---

## 🖥️ Streamlit UI Features

| Page | Features |
|------|---------|
| 🏠 Home & Prediction | Input form, risk gauge, probability bar, business insights |
| 📊 EDA Dashboard | Interactive plots, correlation heatmap, feature explorer |
| 📈 Model Performance | Comparison table, radar chart, ROC curves, confusion matrices |
| ℹ️ About | Project documentation |

---

## 🎛️ Risk Threshold Tuning

The app includes a **decision threshold slider** (default: 0.50).

- **Lower threshold** → Stricter (flag more applicants as risky) → Higher Recall
- **Higher threshold** → Lenient (approve more applicants) → Lower Recall

This allows the bank to tune the system to their risk appetite.

---

## 📦 Requirements

```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
imbalanced-learn==0.12.3
matplotlib==3.8.4
seaborn==0.13.2
shap==0.45.0
streamlit==1.34.0
plotly==5.22.0
joblib==1.4.2
scipy==1.13.0
```

---

## 🏗️ Architecture

```
Raw Data
   │
   ▼
Data Loader (data_loader.py)
   │  5,000 rows, 18 features
   ▼
Preprocessing (preprocessing.py)
   │  Impute → Encode → Scale → SMOTE
   ▼
Training (train.py)
   │  LR | DT | RF | XGBoost
   ▼
Evaluation (evaluate.py)
   │  Acc | Prec | Rec | F1 | AUC
   ▼
Streamlit App (app/app.py)
   │  User Input → Preprocess → Predict → Display
   ▼
Prediction: ✅ Low Risk  /  ⚠️ High Risk
```

---

## 📝 License

MIT License – Free to use for academic and personal projects.

---

*Engineered for Excellence | Credit Risk Analytics Platform | Professional Risk Modeling*
