# Changelog

All notable user-facing changes to `pi-oplsda` are documented here.

## [1.1.0] - 2026-07-02

### Added

- Added a scikit-learn style estimator interface for `OPLSDA`, including
  `fit`, `predict`, `decision_function`, `transform`, `fit_transform`, and
  `score`.
- Added `compatibility="ropls"` and `compatibility="sklearn"` modes to support
  both ropls-aligned statistical reporting and modern Python ML workflows.
- Added automatic orthogonal component selection via `n_ortho="auto"`.
- Added `OPLSDAClassifier` as an alias for `OPLSDA`.
- Added `load_sacurine()` for loading the bundled benchmark dataset.

### Changed

- Updated the default examples to use ropls-compatible automatic orthogonal
  component selection.
- Updated the R benchmark workflow to compare Python `n_ortho="auto"` with R
  `orthoI=NA`; the Sacurine benchmark now selects two orthogonal components in
  both implementations.
- Simplified permutation-test progress reporting to use `tqdm` only.
- Updated diagnostic plot legends and default visualizer palette to match the
  red/gray style used by `pi-metaboqc`.
- Refreshed package metadata, README, and benchmark figure for the 1.1.0 API.

### Fixed

- Improved compatibility with newer scikit-learn validation APIs.
- Improved Windows/R/rpy2 benchmark setup for standalone R installations.
- Adjusted pytest artifact handling to avoid Windows system temporary directory
  permission issues.

## [1.0.3] - 2026-04-13

- Added PyPI publishing workflow and package metadata updates.
- Updated project documentation after the initial 1.0 release.

## [1.0.0] - 2026-04-13

- Initial major release of `pi-oplsda`.
- Included the core OPLS-DA workflow, parallel permutation testing, diagnostic
  visualizations, example notebooks, and ropls comparison documentation.
