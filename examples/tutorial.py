# examples/tutorial.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import piopls
from piopls import OPLSDA, OPLSDA_Visualizer
from piopls import load_sacurine

def main():
    print("=" * 50)
    print(f"   pi-oplsda Terminal Tutorial (Version: {piopls.__version__})")
    print("=" * 50)

    # 1. Load the built-in benchmark dataset (Sacurine)
    print("\n[1/4] Loading benchmark dataset (sacurine)...")
    X, y, feature_names, sample_names = load_sacurine()
    print(f"      - Feature matrix X shape: {X.shape}")
    print(f"      - Target array y shape: {y.shape}")
    print(f"      - Unique classes in y: {np.unique(y)}")

    # 2. Configure and fit the OPLS-DA model
    print("\n[2/4] Fitting OPLS-DA model & computing Q2...")
    model_opls = OPLSDA(
        cv_folds=7,
        max_ortho=10, 
        n_perms=100, 
        n_jobs=-1
    )
    model_opls.fit(X, y)
    model_opls.compute_q2(X, y) 

    # 3. Execute the permutation test 
    # (This will automatically trigger the 'rich' progress bar in terminal)
    print("\n[3/4] Executing permutation test (watch the rich progress bar)...\n")
    perm_results = model_opls.permutation_test(X, y)
    
    # Print a brief overview of the model metrics
    print("\n--- Global Model Overview ---")
    # Using .to_string(index=False) ensures cleaner table formatting in the CLI
    print(model_opls.get_model_info_df().to_string(index=False))
    
    print("\n--- Step-wise Summary ---")
    print(model_opls.get_summary_df().to_string(index=False))

    # 4. Generate and save diagnostic visualizations
    print("\n[4/4] Generating publication-ready visualizations...")
    vis_opls = OPLSDA_Visualizer(
        model=model_opls, 
        y=y, 
        feature_names=feature_names, 
        sample_names=sample_names, 
        vip_threshold=1.0, 
        top_n_vip=15
    )

    # Render the comprehensive plot, and save the figure as a high-resolution 
    # image (safest approach for terminal)
    output_filename = "pi-oplsda_terminal_output.png"
    vis_opls.plot_all(perm_results=perm_results,save_path=None)
    
    print(f"\n[SUCCESS] Pipeline completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()