# π-OPLS-DA (`pi-oplsda`)

[![GitHub release](https://img.shields.io/github/v/release/KaikunXu/pi-oplsda)](https://github.com/KaikunXu/pi-oplsda/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


> A high-performance, Pythonic implementation of Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA), tailored for metabolomics and bioinformatics.

`pi-oplsda` bridges the gap between the rigorous algorithmic foundation of the gold-standard R package `ropls` and the modern Python data science ecosystem. It delivers blazing-fast parallel computing, native Pandas integration, and publication-ready visualizations—all in one lightweight package.

##  Core Capabilities

+ **Standardized Rigor:** Perfectly replicates `ropls` step-wise variance increments ($R^2X$, $R^2Y$, $Q^2$) and NIPALS-based orthogonal signal correction (OSC).
+ **Pandas Native:** Seamlessly feed `pandas.DataFrame` into the model. Sample IDs and feature names are automatically tracked, eliminating the need for tedious matrix index management.
+ **Multi-Core Acceleration:** Powered by `joblib`, permutation tests are fully parallelized, reducing computation time for 100+ permutations from minutes to mere seconds.
+ **Publication-Ready Graphics:** Built on `matplotlib` and `seaborn` to generate clean, high-resolution diagnostic plots with smart legend placement. 
+ **Structured Export:** Extract model parameters, sample scores, and biomarker statistics (VIP, Covariance, Correlation) as instantly usable DataFrames for downstream pipelines.

##  Installation
You can install `pi-oplsda` using either of the following methods, depending on whether you simply want to use the package or if you plan to modify the source code.

Option 1: Install directly from GitHub (Recommended for most users)

```bash
pip install git+https://github.com/KaikunXu/pi-oplsda.git
```

Option 2: Install from source (For developers)

If you want to contribute to the project, modify the algorithm, or explore the source code, you can clone the repository and install it in "editable" mode. This means any changes you make to the local code will immediately take effect without needing to reinstall the package.

```bash
# 1. Clone the repository
git clone https://github.com/KaikunXu/pi-oplsda.git

# 2. Navigate into the project directory
cd pi-oplsda

# 3. Install in editable mode
pip install -e .
```

##  Quickstart & Tutorials
We provide interactive Jupyter Notebooks that walk you through the entire OPLS-DA workflow, from data loading and model fitting to permutation testing and publication-ready visualization.

Instead of static code snippets, please refer to our executable tutorials to get started immediately:

*  **[Quickstart Tutorial (English)](examples/quickstart_en.ipynb)**
*  **[快速入门指南 (中文)](examples/quickstart_zh.ipynb)**

##  Generated Diagnostic Plots

<p align="center">
  <img src="assets/pi-oplsda_visualizer.png" width="1000" alt="pi-oplsda Dashboard">
</p>


Running the visualizer as demonstrated in the notebooks will automatically generate the following tightly integrated subplots:

+ Model Overview: Displays valid $R^2Y$ and $Q^2$ for the predictive ($p_1$) and orthogonal ($o_n$) components.
+ X-Score Plot: Visualizes sample clustering with 95% confidence ellipses.
+ Permutation Test: Validates model robustness against overfitting with empirical p-values.
+ S-Plot: Highlights biomarkers based on covariance and correlation.
+ VIP Bar Plot: Ranks top features contributing to the group separation, featuring **automatic text wrapping** for excessively long metabolite names on the Y-axis to ensure clean and publication-ready layouts.

## Acknowledgements

The algorithmic foundation of `pi-oplsda` is deeply inspired by the excellent R package [`ropls`](https://bioconductor.org/packages/ropls/).

> **Note:** Portions of this codebase, including code refactoring and documentation, were refined with the assistance of Gemini 3.1 Pro. All AI-assisted contributions have been strictly reviewed by the human author to ensure scientific accuracy and code quality.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/KaikunXu/pi-oplsda/issues).

## License

This project is licensed under the **MIT License**.
