"""Built-in dataset loaders for pi-oplsda."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_sacurine() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the bundled Sacurine human urine metabolomics benchmark dataset."""
    module_path = Path(__file__).resolve().parent
    x_path = module_path / "data" / "sacurine_X.csv"
    y_path = module_path / "data" / "sacurine_Y.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Sacurine dataset files not found. Ensure 'data/' folder contains "
            ".csv files."
        )

    # Use index_col=0 to set 'sample_id' column as DataFrame index
    df_X = pd.read_csv(x_path, index_col=0)
    df_y = pd.read_csv(y_path, index_col=0)

    # Assume that the sample IDs in X and Y are the same and in the same order
    if not (df_X.index == df_y.index).all():
        raise ValueError(
            "Critical Error: Sample IDs in X and Y do not match or are out "
            "of order!"
        )

    # Extract feature matrix X (remove sample ID column)
    X = df_X.values
    # Extract target vector (Gender) from the first column of Y
    y = df_y.iloc[:, 0].values

    feature_names = df_X.columns.tolist()
    sample_names = df_X.index.tolist()

    return X, y, feature_names, sample_names
