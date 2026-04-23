"""
pipeline.py
===========
Master pipeline script - runs every step in sequence:

  Step 1 -> Generate / load dataset
  Step 2 -> EDA (plots)
  Step 3 -> Preprocessing (impute, encode, scale)
  Step 4 -> Train all models
  Step 5 -> Evaluate & compare
  Step 6 -> Save best model

Run:
    python src/pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from data_loader   import load_data
from preprocessing import preprocess
from train         import train_all
from eda           import run_eda
from evaluate      import evaluate_all

import joblib


def main():
    banner = "=" * 60
    print()
    print(banner)
    print("   CREDIT RISK ASSESSMENT - FULL PIPELINE")
    print(banner)
    print()
    total_start = time.time()

    # --- Step 1: Data ---------------------------------------------------------
    print(">> Step 1/5 : Loading dataset ...")
    df = load_data()
    print("   Rows: {:,}   |   Columns: {}".format(len(df), df.shape[1]))
    print()

    # --- Step 2: EDA ----------------------------------------------------------
    print(">> Step 2/5 : Running EDA ...")
    run_eda(df)
    print()

    # --- Step 3: Preprocessing ------------------------------------------------
    print(">> Step 3/5 : Preprocessing (impute -> encode -> scale) ...")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df, fit=True)
    print("   Train: {}   |   Test: {}".format(X_train.shape, X_test.shape))
    print()

    # --- Step 4: Training -----------------------------------------------------
    print(">> Step 4/5 : Training models ...")
    trained_models = train_all(X_train, y_train)
    print()

    # --- Step 5: Evaluation ---------------------------------------------------
    print(">> Step 5/5 : Evaluating & saving best model ...")
    evaluate_all()

    # --- Done -----------------------------------------------------------------
    elapsed = time.time() - total_start
    print()
    print(banner)
    print("   Pipeline complete in {:.1f}s".format(elapsed))
    print("   [OK] Models      -> models/")
    print("   [OK] EDA plots   -> reports/")
    print("   [OK] Eval plots  -> reports/")
    print()
    print("   Launch the Streamlit app:")
    print("       streamlit run app/app.py")
    print(banner)
    print()


if __name__ == "__main__":
    main()
