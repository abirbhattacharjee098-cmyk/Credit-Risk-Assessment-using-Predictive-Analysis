"""
preprocessing.py
================
Full data-preprocessing pipeline for the Credit Risk Assessment project.

Steps performed:
  1. Drop duplicate rows
  2. Impute missing numeric values (median strategy)
  3. Encode categorical features (one-hot encoding)
  4. Scale numeric features (StandardScaler)
  5. Train/test split
  Note: Class imbalance is handled via class_weight='balanced' in each model.

Usage (standalone):
    python src/preprocessing.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import StandardScaler
from sklearn.impute              import SimpleImputer

# -- local imports -------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data

# -- constants -----------------------------------------------------------------
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TARGET_COL     = "default"
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "..", "models")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH   = os.path.join(MODELS_DIR, "imputer.pkl")
COLUMNS_PATH   = os.path.join(MODELS_DIR, "feature_columns.pkl")

NUMERIC_FEATURES = [
    "age", "income", "loan_amount", "loan_term",
    "credit_score", "employment_years", "num_credit_lines",
    "debt_to_income", "num_delinquencies", "previous_defaults",
    "loan_to_income", "monthly_payment", "payment_to_income",
]

CATEGORICAL_FEATURES = [
    "education", "home_ownership", "loan_purpose", "employment_status",
]


# -----------------------------------------------------------------------------
def preprocess(df: pd.DataFrame,
               fit: bool = True,
               apply_smote: bool = False):
    """
    Preprocess the raw credit DataFrame.

    Parameters
    ----------
    df          : raw DataFrame returned by load_data()
    fit         : if True, fit scaler/imputer and save them to disk.
                  Set to False when transforming new (inference) data.
    apply_smote : legacy param kept for API compatibility (not used).

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -- 1. Drop duplicates ------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocess] Dropped {before - len(df)} duplicate rows.")

    # -- 2. Separate target ------------------------------------------------
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # -- 3. Impute missing numeric values ----------------------------------
    if fit:
        imputer = SimpleImputer(strategy="median")
        X[NUMERIC_FEATURES] = imputer.fit_transform(X[NUMERIC_FEATURES])
        joblib.dump(imputer, IMPUTER_PATH)
        print(f"[preprocess] Imputer saved -> {IMPUTER_PATH}")
    else:
        imputer = joblib.load(IMPUTER_PATH)
        X[NUMERIC_FEATURES] = imputer.transform(X[NUMERIC_FEATURES])

    # -- 4. One-hot encode categorical columns -----------------------------
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)

    # -- 5. Align columns (inference mode may be missing some dummies) -----
    if fit:
        feature_names = X.columns.tolist()
        joblib.dump(feature_names, COLUMNS_PATH)
        print(f"[preprocess] Feature columns saved -> {COLUMNS_PATH}")
    else:
        feature_names = joblib.load(COLUMNS_PATH)
        # Add any missing columns (all zeros) and reorder
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

    # -- 6. Train / test split ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[preprocess] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"[preprocess] Class distribution (train) -> "
          f"0: {(y_train==0).sum()}  |  1: {(y_train==1).sum()}")

    # -- 7. Scale numeric features -----------------------------------------
    if fit:
        scaler = StandardScaler()
        X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocess] Scaler saved -> {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)

    # Scale test set with already-fitted scaler
    numeric_cols_in_X = [c for c in NUMERIC_FEATURES if c in X_test.columns]
    X_test[numeric_cols_in_X] = scaler.transform(X_test[numeric_cols_in_X])

    # -- 8. Class imbalance handled via class_weight='balanced' in each model --
    # (SMOTE removed: requires imbalanced-learn not available in this env)

    return X_train, X_test, y_train, y_test, feature_names


def preprocess_single(input_dict: dict) -> np.ndarray:
    """
    Transform a single applicant's data (dict) into a model-ready array.
    Used by the Streamlit app and Flask API.

    Parameters
    ----------
    input_dict : {feature_name: value}  (raw / un-scaled input from the UI)

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    df_single = pd.DataFrame([input_dict])

    # Impute
    imputer = joblib.load(IMPUTER_PATH)
    df_single[NUMERIC_FEATURES] = imputer.transform(
        df_single[NUMERIC_FEATURES].values
    )

    # One-hot encode
    df_single = pd.get_dummies(df_single, columns=CATEGORICAL_FEATURES,
                               drop_first=False)

    # Align columns
    feature_names = joblib.load(COLUMNS_PATH)
    for col in feature_names:
        if col not in df_single.columns:
            df_single[col] = 0
    df_single = df_single[feature_names]

    # Scale numeric features
    scaler = joblib.load(SCALER_PATH)
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df_single.columns]
    df_single[numeric_cols] = scaler.transform(df_single[numeric_cols].values)

    return df_single.values


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)
    print(f"\nFinal X_train shape : {X_train.shape}")
    print(f"Final X_test  shape : {X_test.shape}")
    print(f"Total features      : {len(features)}")
