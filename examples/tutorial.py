# examples/tutorial.py
"""Terminal tutorial for the pi-oplsda module.

Demonstrates the primary metabolomics workflow: fitting the entire dataset 
for VIP-based feature extraction and validation, followed by a demonstration 
of Train/Test predictive capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import piopls
from piopls import OPLSDA, OPLSDA_Visualizer
from piopls import load_sacurine


def main():
    """Executes the complete pi-oplsda workflow in the terminal."""
    print("=" * 60)
    print(f"   pi-oplsda Terminal Tutorial (Version: {piopls.__version__})")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Full Dataset Modeling & Feature Extraction
    # ---------------------------------------------------------
    print("\n[1/4] Loading dataset and fitting global OPLS-DA model...")
    X, y_data, feature_names, sample_names = load_sacurine()
    
    model_opls = OPLSDA(
        n_ortho=1,
        cv_folds=7,
        cv_strategy="stratified",
        compatibility="sklearn",
        random_state=42,
        n_perms=100,
        n_jobs=-1,
    )
    model_opls.fit(X, y_data)
    model_opls.compute_q2(X, y_data)
    
    print("\n--- Global Model Overview ---")
    df_info = model_opls.get_model_info_df()
    print(df_info.to_string(index=False))

    print("\n--- Top 5 Key Metabolites (Ranked by VIP) ---")
    df_features = model_opls.get_features_df(feature_names=feature_names)
    print(df_features.head(5).to_string(index=False))

    # ---------------------------------------------------------
    # 2. Permutation Testing
    # ---------------------------------------------------------
    print("\n[2/4] Executing permutation test on full dataset...")
    perm_results = model_opls.permutation_test(X, y_data)

    # ---------------------------------------------------------
    # 3. Global Visualization
    # ---------------------------------------------------------
    print("\n[3/4] Generating global diagnostic dashboard...")
    vis_opls = OPLSDA_Visualizer(
        model=model_opls, 
        y=y_data, 
        feature_names=feature_names, 
        sample_names=sample_names, 
        vip_threshold=1.0, 
        top_n_vip=20
    )
    
    vis_opls.plot_all(
        perm_results=perm_results, 
        wrap_width=30, 
        show_plot=True
    )

    # ---------------------------------------------------------
    # 4. Advanced: Train/Test Prediction
    # ---------------------------------------------------------
    print("\n[4/4] Demonstrating predictive capabilities (Train/Test Split)...")
    
    X_train, y_train = X[0::2].copy(), y_data[0::2].copy()
    X_test, y_test = X[1::2].copy(), y_data[1::2].copy()
    
    model_predict = OPLSDA(
        n_ortho=1,
        cv_folds=7,
        cv_strategy="stratified",
        random_state=42,
    )
    model_predict.fit(X_train, y_train)
    
    y_pred_test = model_predict.predict(X_test)

    cm_test = pd.crosstab(
        pd.Series(y_test, name='Actual'), 
        pd.Series(y_pred_test, name='Predicted')
    )

    print("\n--- Testing Set Confusion Matrix ---")
    print(cm_test.to_string())
    
    print("=" * 60)
    print("   Tutorial Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
