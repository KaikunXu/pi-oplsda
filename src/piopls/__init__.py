# src/piopls/__init__.py

"""Python utilities for Orthogonal Partial Least Squares Discriminant Analysis.

The package provides ropls-aligned OPLS-DA computation, parallel permutation
tests, scikit-learn style estimator methods, and publication-ready diagnostic
visualizations.
"""

from .oplsda_models import OPLSDA, OPLSDAClassifier
from .oplsda_plotting import OPLSDA_Visualizer
from .datasets import load_sacurine

__all__ = [
    "OPLSDA",
    "OPLSDAClassifier",
    "OPLSDA_Visualizer",
    "load_sacurine",
]

__version__ = "1.1.1"
