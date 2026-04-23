"""
train.py
========
Trains four ML models on the preprocessed credit-risk dataset:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost

Each model is saved as a .pkl file in the models/ directory.
Best model selection is based on ROC-AUC on the held-out test set.

Usage:
    python src/train.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# -- local imports - works both as main script and as module import -------------
sys.path.insert(0, os.path.dirname(__file__))
from data_loader   import load_data
from preprocessing import preprocess

# -- Paths ---------------------------------------------------------------------
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
def get_models() -> dict:
    """Return a dict of {model_name: sklearn_estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # handles remaining imbalance
            random_state=RANDOM_STATE,
            solver="lbfgs",
            C=0.1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_split=20,
            random_state=RANDOM_STATE,
        ),
    }


# -----------------------------------------------------------------------------
def train_all(X_train, y_train) -> dict:
    """
    Train every model defined in get_models().

    Returns
    -------
    dict  {model_name: fitted_estimator}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models  = get_models()
    trained = {}

    for name, model in models.items():
        print(f"\n{'-'*60}")
        print(f"  Training: {name}")
        print(f"{'-'*60}")
        t0 = time.time()

        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        # 5-fold CV on training data for a quick sanity AUC
        cv_auc = cross_val_score(model, X_train, y_train,
                                 cv=5, scoring="roc_auc", n_jobs=-1)
        print(f"  CV ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        print(f"  Time taken : {elapsed:.1f}s")

        # Persist to disk
        save_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, save_path)
        print(f"  Saved      -> {save_path}")

        trained[name] = model

    return trained


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CREDIT RISK ASSESSMENT - MODEL TRAINING")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    trained_models = train_all(X_train, y_train)

    print("\n" + "=" * 60)
    print("  All models trained and saved successfully!")
    print("  Run  python src/evaluate.py  to see full metrics.")
    print("=" * 60)
