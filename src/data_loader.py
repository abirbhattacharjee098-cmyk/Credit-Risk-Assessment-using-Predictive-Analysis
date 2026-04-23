"""
data_loader.py
==============
Generates and loads a realistic synthetic credit risk dataset based on the
structure of the German Credit Dataset / Lending Club dataset.

The dataset is saved to data/credit_data.csv so every module uses the
same file.  Run this script once before anything else:

    python src/data_loader.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_SAMPLES    = 5000
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "credit_data.csv")


def generate_credit_dataset(n_samples: int = N_SAMPLES,
                             random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Synthesise a realistic credit-risk dataset with known feature semantics.

    Returns
    -------
    pd.DataFrame  - raw (un-processed) dataset ready for EDA / preprocessing.
    """
    rng = np.random.default_rng(random_state)

    # -- Core numeric features ---------------------------------------------
    age             = rng.integers(20, 70, n_samples)
    income          = rng.integers(15_000, 200_000, n_samples).astype(float)
    loan_amount     = rng.integers(1_000,  50_000,  n_samples).astype(float)
    loan_term       = rng.choice([12, 24, 36, 48, 60], n_samples)
    credit_score    = rng.integers(300, 850, n_samples)
    employment_years= rng.integers(0,  30,  n_samples).astype(float)
    num_credit_lines= rng.integers(1,  20,  n_samples)
    debt_to_income  = np.round(rng.uniform(0.05, 0.60, n_samples), 4)
    num_delinquencies = rng.integers(0, 10, n_samples)
    previous_defaults = rng.integers(0,  4, n_samples)

    # -- Categorical features ----------------------------------------------
    education   = rng.choice(["High School", "Bachelor", "Master", "PhD"],
                              n_samples, p=[0.30, 0.45, 0.18, 0.07])
    home_ownership = rng.choice(["Rent", "Own", "Mortgage"],
                                 n_samples, p=[0.35, 0.25, 0.40])
    loan_purpose = rng.choice(
        ["Debt Consolidation", "Home Improvement", "Business",
         "Medical", "Education", "Personal"],
        n_samples, p=[0.30, 0.20, 0.15, 0.10, 0.10, 0.15]
    )
    employment_status = rng.choice(
        ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        n_samples, p=[0.60, 0.15, 0.15, 0.10]
    )

    # -- Derived / engineered columns --------------------------------------
    loan_to_income    = np.round(loan_amount / (income + 1), 4)
    monthly_payment   = np.round(
        (loan_amount * 0.07 / 12) / (1 - (1 + 0.07 / 12) ** -loan_term), 2
    )
    payment_to_income = np.round(monthly_payment / (income / 12 + 1), 4)

    # -- Target: default (1 = default / high risk)  ------------------------
    # We construct a realistic risk score so the label correlates
    # meaningfully with the features.
    risk_score = (
        -0.01  * credit_score
        +  0.5  * debt_to_income
        +  1.5  * previous_defaults
        +  0.3  * num_delinquencies
        +  0.5  * loan_to_income
        - 0.005 * income / 1_000
        +  rng.normal(0, 0.5, n_samples)   # noise
    )
    prob_default = 1 / (1 + np.exp(-risk_score))   # sigmoid -> probability
    default      = (prob_default > 0.55).astype(int) # ~30 % default rate

    # -- Introduce realistic missing values --------------------------------
    missing_mask = rng.random(n_samples) < 0.03   # 3 % missing in income
    income[missing_mask] = np.nan

    missing_mask2 = rng.random(n_samples) < 0.04
    employment_years[missing_mask2] = np.nan

    # -- Assemble DataFrame ------------------------------------------------
    df = pd.DataFrame({
        "age":               age,
        "income":            income,
        "loan_amount":       loan_amount,
        "loan_term":         loan_term,
        "credit_score":      credit_score,
        "employment_years":  employment_years,
        "num_credit_lines":  num_credit_lines,
        "debt_to_income":    debt_to_income,
        "num_delinquencies": num_delinquencies,
        "previous_defaults": previous_defaults,
        "loan_to_income":    loan_to_income,
        "monthly_payment":   monthly_payment,
        "payment_to_income": payment_to_income,
        "education":         education,
        "home_ownership":    home_ownership,
        "loan_purpose":      loan_purpose,
        "employment_status": employment_status,
        "default":           default,          # <- TARGET
    })

    return df


def load_data(path: str = OUTPUT_PATH) -> pd.DataFrame:
    """Load dataset from CSV; generate it first if the file does not exist."""
    if not os.path.exists(path):
        print(f"[data_loader] Dataset not found at '{path}'. Generating …")
        df = generate_credit_dataset()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[data_loader] Saved {len(df):,} rows -> '{path}'")
    else:
        df = pd.read_csv(path)
        print(f"[data_loader] Loaded {len(df):,} rows from '{path}'")
    return df


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print("\nShape      :", df.shape)
    print("Default rate: {:.1%}".format(df["default"].mean()))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
