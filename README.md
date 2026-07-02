# `pi-oplsda`: pi-OPLS-DA

[![PyPI version](https://badgen.net/pypi/v/pi-oplsda)](https://pypi.org/project/pi-oplsda/)
[![License](https://badgen.net/github/license/KaikunXu/pi-oplsda)](https://github.com/KaikunXu/pi-oplsda/blob/main/LICENSE)
[![Python Version](https://badgen.net/pypi/python/pi-oplsda)](https://pypi.org/project/pi-oplsda/)

`pi-oplsda` bridges the rigorous algorithmic foundation of the R package `ropls` with the modern Python data science ecosystem. It provides native Pandas support, parallel permutation testing, scikit-learn style estimator methods, ropls-compatible statistical reporting, and publication-ready visualizations in one lightweight package.

## Core Capabilities

- **ropls-compatible computation:** Reproduces ropls-style OPLS-DA summaries, including cumulative $R^2X$, $R^2Y$, $Q^2$, RMSEE, VIP scores, predictive scores/loadings, and orthogonal scores.
- **Automatic orthogonal component selection:** Set `n_ortho="auto"` to select the number of orthogonal components with ropls-style rules.
- **Two compatibility modes:** Use `compatibility="ropls"` for R comparison and visualization consistency, or `compatibility="sklearn"` for fold-local preprocessing and modern Python ML conventions.
- **Scikit-learn style API:** `fit`, `predict`, `decision_function`, `transform`, `fit_transform`, and `score` are available on `OPLSDA`.
- **Pandas native:** `pandas.DataFrame` inputs preserve sample IDs and feature names in exported result tables.
- **Multi-core acceleration:** Permutation tests are parallelized with `joblib` and reported with `tqdm`.
- **Publication-ready graphics:** `OPLSDA_Visualizer` generates diagnostic plots for model summary, scores, outliers, permutation tests, VIP rankings, and S-plot style feature statistics.
- **Structured export:** Model metadata, component summaries, sample scores, and feature statistics are available as DataFrames.

> **Note:** Due to the nature of latent-variable models, scores and loadings may be flipped in sign between platforms. This is mathematically equivalent and does not affect interpretation.

## Installation

Install from PyPI:

```bash
pip install pi-oplsda
```

Install from GitHub:

```bash
pip install git+https://github.com/KaikunXu/pi-oplsda.git
```

Install from source for development:

```bash
git clone https://github.com/KaikunXu/pi-oplsda.git
cd pi-oplsda
pip install -e ".[dev]"
```

For benchmark/testing against R `ropls`, install the Python-side optional dependencies:

```bash
pip install -e ".[test]"
```

The R package `ropls` should be installed in your regular R installation. It is not bundled with this Python package.

## Quickstart & Tutorials

Interactive examples are provided in the `examples` directory:

- **[Quickstart Tutorial](https://github.com/KaikunXu/pi-oplsda/blob/main/examples/quickstart.ipynb):** ropls-compatible workflow with automatic orthogonal component selection.
- **[R-ropls Equivalence Benchmark](https://github.com/KaikunXu/pi-oplsda/blob/main/examples/benchmark.ipynb):** numerical comparison between Python and R implementations.
- **[Terminal Tutorial](https://github.com/KaikunXu/pi-oplsda/blob/main/examples/tutorial.py):** a script-oriented workflow.

Minimal ropls-compatible usage:

```python
from piopls import OPLSDA, load_sacurine

X, y, feature_names, sample_names = load_sacurine()

model = OPLSDA(
    n_ortho="auto",
    max_ortho=10,
    cv_folds=7,
    compatibility="ropls",
    random_state=42,
)

model.fit_pipeline(X, y, run_permutations=False)

print(model.n_ortho_)
print(model.get_model_info_df())
print(model.get_summary_df())
```

For a scikit-learn style workflow, use the default `compatibility="sklearn"` or set it explicitly:

```python
model = OPLSDA(n_ortho=1, compatibility="sklearn", random_state=42)
model.fit(X, y)

y_pred = model.predict(X)
y_score = model.decision_function(X)
accuracy = model.score(X, y)
```

## Visualization

Running `OPLSDA_Visualizer` generates a suite of diagnostic subplots:

- **Model Overview:** Step-wise and cumulative model quality metrics.
- **X-Score Plot:** Sample clustering in predictive and orthogonal latent spaces with confidence ellipses.
- **Observation Diagnostics:** Score distance and orthogonal distance / DModX for outlier inspection.
- **Permutation Test:** Original $R^2Y$ and $Q^2$ against permuted null distributions.
- **VIP Bar Plot:** Top features contributing to group separation.
- **S-Plot:** Feature covariance and correlation diagnostics for binary models.

```python
from piopls import OPLSDA_Visualizer

vis = OPLSDA_Visualizer(
    model=model,
    y=y,
    feature_names=feature_names,
    sample_names=sample_names,
    vip_threshold=1.0,
    top_n_vip=20,
)

vis.plot_all()
```

![pi-oplsda_visualizer](https://raw.githubusercontent.com/KaikunXu/pi-oplsda/main/assets/pi-oplsda_visualizer.png)

## Mathematical Equivalence & Benchmarking

`pi-oplsda` is validated against the R/Bioconductor package `ropls` to ensure scientific consistency. The benchmark uses the Sacurine human urine metabolomics dataset with 183 samples and 109 metabolites.

The current benchmark aligns the two implementations as follows:

- Python: `OPLSDA(n_ortho="auto", compatibility="ropls")`
- R: `ropls::opls(..., predI = 1, orthoI = NA, crossvalI = 7)`

Both implementations select two orthogonal components on the Sacurine dataset.

| Metric | Description | Comparison |
| :--- | :--- | :--- |
| **Global Quality** | Cumulative $R^2X$, $R^2Y$, and $Q^2$ | Approximately equal |
| **Error Assessment** | Root Mean Square Error of Estimation (RMSEE) | Approximately equal |
| **Latent Space** | Predictive scores, predictive loadings, and orthogonal scores | Pearson's r ~= 1 |
| **Variable Importance** | Variable Importance in Projection (VIP) scores | Pearson's r ~= 1 |

The benchmark figure summarizes global metrics and five vector-level comparisons: predictive scores (`t1`), predictive loadings (`p1`), VIP scores, and two orthogonal score vectors (`to1`, `to2`).

![pi_oplsda_benchmark.png](https://raw.githubusercontent.com/KaikunXu/pi-oplsda/main/assets/pi_oplsda_benchmark.png)

## Testing

Run the test suite with:

```bash
pytest tests
```

The R benchmark test uses `rpy2` and skips cleanly when a standalone R installation or the R package `ropls` is unavailable. On Windows, the benchmark setup expects a regular R installation such as `D:/R/R-4.5.2`.

## Changelog

Release notes are maintained separately in [CHANGELOG.md](https://github.com/KaikunXu/pi-oplsda/blob/main/CHANGELOG.md).

## Acknowledgements

The algorithmic foundation of `pi-oplsda` is inspired by the excellent R package [`ropls`](https://bioconductor.org/packages/ropls/).

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/KaikunXu/pi-oplsda/issues).

## License

This project is licensed under the **MIT License**.
