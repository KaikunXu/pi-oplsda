# src/piopls/__init__.py

"""
piopls: A Python package for Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA).
Strictly aligned with ropls algorithm definitions, providing highly efficient parallel 
permutation tests and publication-ready visualizations.
"""

from .oplsda_models import OPLSDA
from .oplsda_plotting import OPLSDA_Visualizer
from .datasets import load_sacurine

__all__ = [
    "OPLSDA",
    "OPLSDA_Visualizer",
    "load_sacurine"
]

__version__ = "1.0.2"