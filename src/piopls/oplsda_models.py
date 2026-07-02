"""Orthogonal Partial Least Squares Discriminant Analysis models.

The public ``OPLSDA`` estimator follows the scikit-learn classifier API while
keeping the ropls-style summary tables used by the plotting layer.  The default
validation behavior is intentionally sklearn-like: preprocessing is learned
inside each cross-validation fold.  ropls-compatible choices are exposed through
explicit parameters rather than hidden defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm


@dataclass(frozen=True)
class _PreparedX:
    values: np.ndarray
    feature_names: list[str] | None
    sample_names: list[str] | None


class OPLSDA(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible binary OPLS-DA estimator.

    Parameters
    ----------
    n_ortho:
        Number of orthogonal components. Use ``"auto"`` or ``None`` to select
        the component count by cross-validated Q2 rules.
    cv_folds:
        Number of folds used by :meth:`compute_q2`.
    compatibility:
        ``"sklearn"`` keeps modern ML defaults. ``"ropls"`` switches the
        default CV splitter, p-value denominator, RMSEE correction, and auto
        component rule toward ropls conventions.
    cv_strategy:
        ``"auto"``, ``"stratified"``, ``"kfold"``, ``"ropls_venetian"``, or
        ``"stratified_venetian"``. A splitter object with ``split`` is also
        accepted.
    cv_preprocess:
        ``"fold"`` estimates scaling inside each fold. ``"global"`` reuses the
        full-data scaling parameters during Q2 computation for ropls-style
        comparisons.
    scale:
        Predictor scaling: ``"standard"``, ``"center"``, ``"pareto"``, or
        ``"none"``.
    """

    def __init__(
        self,
        n_ortho=1,
        cv_folds=7,
        max_ortho=10,
        n_perms=100,
        n_jobs=-1,
        vip_method="vip4",
        *,
        compatibility="sklearn",
        cv_strategy="auto",
        cv_preprocess="fold",
        scale="standard",
        selection_rule="auto",
        auto_q2_min_delta=0.01,
        remove_zero_variance=True,
        missing="error",
        shuffle=False,
        random_state=None,
        pvalue_method="auto",
        rmsee_correction="auto",
        allow_multiclass=False,
    ):
        self.n_ortho = n_ortho
        self.cv_folds = cv_folds
        self.max_ortho = max_ortho
        self.n_perms = n_perms
        self.n_jobs = n_jobs
        self.vip_method = vip_method
        self.compatibility = compatibility
        self.cv_strategy = cv_strategy
        self.cv_preprocess = cv_preprocess
        self.scale = scale
        self.selection_rule = selection_rule
        self.auto_q2_min_delta = auto_q2_min_delta
        self.remove_zero_variance = remove_zero_variance
        self.missing = missing
        self.shuffle = shuffle
        self.random_state = random_state
        self.pvalue_method = pvalue_method
        self.rmsee_correction = rmsee_correction
        self.allow_multiclass = allow_multiclass

    # ------------------------------------------------------------------
    # Public sklearn-style API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the OPLS-DA classifier."""
        self._validate_params()
        x_prepared = self._prepare_X(X, reset=True)
        X_arr = x_prepared.values
        y_arr = np.asarray(y)

        if y_arr.ndim != 1:
            y_arr = np.ravel(y_arr)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if pd.isna(y_arr).any():
            raise ValueError("OPLSDA does not accept missing class labels.")

        self.label_encoder = LabelEncoder()
        y_num = self.label_encoder.fit_transform(y_arr).astype(float)
        self.classes_ = self.label_encoder.classes_
        self._is_categorical = True

        if len(self.classes_) != 2 and not self.allow_multiclass:
            raise ValueError(
                "OPLSDA is defined as a binary classifier. Use two classes, "
                "or set allow_multiclass=True for experimental ordinal coding."
            )

        self.feature_names_in_ = x_prepared.feature_names
        self.sample_names_in_ = x_prepared.sample_names
        self.n_features_in_ = X_arr.shape[1]

        X_model = self._apply_zero_variance_filter(X_arr, fit=True)
        self.n_model_features_in_ = X_model.shape[1]
        self.kept_feature_names_ = self._kept_feature_names()

        n_ortho = self._resolve_n_ortho(X_model, y_num)
        self.n_ortho_ = n_ortho

        self._fit_numeric(X_model, y_num)
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        check_is_fitted(self, "classes_")
        y_pred_num = self._predict_continuous(X)
        if len(self.classes_) == 2:
            pred_idx = (y_pred_num > 0.5).astype(int)
        else:
            pred_idx = np.clip(
                np.round(y_pred_num).astype(int), 0, len(self.classes_) - 1
            )
        return self.label_encoder.inverse_transform(pred_idx)

    def decision_function(self, X):
        """Return the continuous OPLS-DA response before thresholding."""
        return self._predict_continuous(X)

    def transform(self, X):
        """Project X onto predictive and orthogonal score components."""
        scores = self._project_scores(X)
        if self.n_ortho_ > 0:
            return np.column_stack([scores["t_pred"], scores["t_ortho"]])
        return scores["t_pred"][:, np.newaxis]

    def fit_transform(self, X, y):
        """Fit the estimator and return training scores."""
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        """Return classification accuracy, matching sklearn classifier API."""
        return accuracy_score(y, self.predict(X))

    # ------------------------------------------------------------------
    # ropls-style workflow helpers retained for existing users
    # ------------------------------------------------------------------
    def fit_pipeline(self, X, y, run_permutations=True) -> dict:
        """Fit, compute Q2, and optionally run a permutation test."""
        self.fit(X, y)
        self.compute_q2(X, y)
        if run_permutations:
            return self.permutation_test(X, y)
        return {}

    def compute_q2(self, X, y):
        """Compute cross-validated Q2 values for 0..n_ortho components."""
        check_is_fitted(self, "classes_")
        x_prepared = self._prepare_X(X, reset=False)
        X_arr = self._apply_zero_variance_filter(x_prepared.values, fit=False)
        y_num = self.label_encoder.transform(np.asarray(y)).astype(float)

        n_ortho_plus_1 = self.n_ortho_ + 1
        y_cv_preds = {
            n: np.zeros(y_num.shape[0], dtype=float) for n in range(n_ortho_plus_1)
        }

        for train_idx, test_idx in self._iter_cv_splits(X_arr, y_num):
            X_train, y_train = X_arr[train_idx], y_num[train_idx]
            X_test = X_arr[test_idx]

            for n in range(n_ortho_plus_1):
                model_cv = self._new_like(n_ortho=n)
                model_cv.label_encoder = self.label_encoder
                model_cv.classes_ = self.classes_
                model_cv._is_categorical = True
                model_cv.feature_names_in_ = self.kept_feature_names_
                model_cv.sample_names_in_ = None
                model_cv.n_features_in_ = X_train.shape[1]
                model_cv.zero_variance_mask_ = np.ones(X_train.shape[1], dtype=bool)
                model_cv.n_model_features_in_ = X_train.shape[1]
                model_cv.kept_feature_names_ = self.kept_feature_names_
                model_cv.n_ortho_ = n

                if self._effective_cv_preprocess() == "global":
                    model_cv._fit_numeric(
                        X_train,
                        y_train,
                        x_mean=self.x_mean_,
                        x_std=self.x_std_,
                        y_mean=self.y_mean_,
                        y_std=self.y_std_,
                    )
                else:
                    model_cv._fit_numeric(X_train, y_train)
                y_cv_preds[n][test_idx] = model_cv._predict_continuous_prepared(
                    X_test
                )

        ss_y = np.nansum((y_num - np.nanmean(y_num)) ** 2)
        press_list = [
            np.nansum((y_num - y_cv_preds[n]) ** 2) for n in range(n_ortho_plus_1)
        ]

        self.Q2_abs_ = [1.0 - (press / ss_y) for press in press_list]
        self.Q2_comp_ = []
        self.Q2_cum_list_ = []
        for i, q2_abs in enumerate(self.Q2_abs_):
            if i == 0:
                self.Q2_comp_.append(q2_abs)
                self.Q2_cum_list_.append(q2_abs)
            else:
                self.Q2_comp_.append(q2_abs - self.Q2_abs_[i - 1])
                self.Q2_cum_list_.append(q2_abs - self.Q2_abs_[0])

        self.Q2_ = self.Q2_abs_[-1]
        self.q2_ = self.Q2_
        return self.Q2_

    def permutation_test(self, X, y, n_perms=None, n_jobs=None):
        """Run a response-label permutation test."""
        check_is_fitted(self, "classes_")
        n_perms = int(n_perms or self.n_perms)
        n_jobs = self.n_jobs if n_jobs is None else n_jobs

        x_prepared = self._prepare_X(X, reset=False)
        X_arr = self._apply_zero_variance_filter(x_prepared.values, fit=False)
        y_num = self.label_encoder.transform(np.asarray(y)).astype(float)

        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, np.iinfo(np.int32).max, size=n_perms)

        results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(self._single_permutation)(X_arr, y_num, int(seed))
            for seed in seeds
        )

        pbar = tqdm(
            results_gen,
            total=n_perms,
            desc="Permutation Test",
            ncols=80,
            colour="#2CA02C",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA: {remaining}]",
            leave=True,
        )
        results = list(pbar)

        perms_r2y = [res[0] for res in results]
        perms_q2 = [res[1] for res in results]
        valid_perms = len(perms_r2y)
        if valid_perms == 0:
            raise ValueError("All permutation iterations failed to converge.")

        orig_r2y = self.R2Y_abs_[-1]
        if hasattr(self, "Q2_"):
            orig_q2 = self.Q2_
        else:
            orig_q2 = self.compute_q2(X, y)
        count_r2y = sum(1 for val in perms_r2y if val >= orig_r2y)
        count_q2 = sum(1 for val in perms_q2 if val >= orig_q2)

        denom = valid_perms if self._effective_pvalue_method() == "ropls" else valid_perms + 1
        p_r2y = (count_r2y + 1) / denom
        p_q2 = (count_q2 + 1) / denom

        self.p_R2Y_ = p_r2y
        self.p_Q2_ = p_q2

        return {
            "orig_R2Y": orig_r2y,
            "orig_Q2Y": orig_q2,
            "perm_R2Y": perms_r2y,
            "perm_Q2Y": perms_q2,
            "p_R2Y": p_r2y,
            "p_Q2Y": p_q2,
            "valid_perms": valid_perms,
        }

    # ------------------------------------------------------------------
    # Export helpers used by the visualizer and notebooks
    # ------------------------------------------------------------------
    def get_model_info_df(self):
        """Return one-row model metadata in the ropls summary style."""
        if not hasattr(self, "Q2_"):
            raise ValueError("Model must compute Q2 first.")

        data = {
            "N_Predictive": [1],
            "N_Ortho": [self.n_ortho_],
            "R2X(cum)": [self.R2X_cum_list_[-1]],
            "R2Y(cum)": [self.R2Y_abs_[-1]],
            "Q2(cum)": [self.Q2_abs_[-1]],
            "RMSEE": [self.RMSEE_],
            "Compatibility": [self.compatibility],
            "CV_Strategy": [self._effective_cv_strategy_name()],
        }
        if hasattr(self, "p_R2Y_"):
            data["pR2Y"] = [self.p_R2Y_]
            data["pQ2"] = [self.p_Q2_]
        return pd.DataFrame(data)

    def get_summary_df(self):
        """Return component-wise R2X/R2Y/Q2 values."""
        if not hasattr(self, "Q2_comp_"):
            raise ValueError("Model must compute Q2 first.")

        components = ["Predictive (p1)"] + [
            f"Orthogonal (o{i + 1})" for i in range(self.n_ortho_)
        ]
        return pd.DataFrame(
            {
                "Component": components,
                "R2X": self.R2X_comp_,
                "R2Y": self.R2Y_comp_,
                "Q2": self.Q2_comp_,
                "R2X(cum)": self.R2X_cum_list_,
                "R2Y(cum)": self.R2Y_cum_list_,
                "Q2(cum)": self.Q2_cum_list_,
            }
        )

    def get_scores_df(self, sample_names=None, y_true=None):
        """Return sample scores and fitted labels."""
        names = self._resolve_sample_names(sample_names)
        data = {"Sample": names, "t_pred (p1)": self.t_pred_.flatten()}
        for i in range(self.n_ortho_):
            data[f"t_ortho (o{i + 1})"] = self.T_ortho_[:, i]
        if y_true is not None:
            data["True_Class"] = np.asarray(y_true)
        if hasattr(self, "fitted_values_"):
            data["Fitted_Value"] = self.fitted_values_.flatten()
        if hasattr(self, "fitted_class_"):
            data["Fitted_Class"] = self.fitted_class_.flatten()

        df = pd.DataFrame(data)
        if "True_Class" in df.columns and "Fitted_Class" in df.columns:
            df["Match_Status"] = np.where(
                df["True_Class"] == df["Fitted_Class"], "Matched", "Mismatched"
            )
        return df

    def get_features_df(self, feature_names=None):
        """Return feature-level VIP, covariance, and correlation metrics."""
        names = self._resolve_feature_names(feature_names)
        df = pd.DataFrame(
            {
                "Feature": names,
                "VIP": self.vip_ropls_,
                "Covariance (p1)": self.covariances_,
                "Correlation (pcorr1)": self.correlations_,
                "Loading_Weight": self.p_pred_.flatten(),
            }
        )
        return df.sort_values(by="VIP", ascending=False).reset_index(drop=True)

    def get_outlier_df(self, sample_names=None, y_true=None):
        """Return score/orthogonal distances and threshold flags."""
        names = self._resolve_sample_names(sample_names)
        df = pd.DataFrame(
            {
                "Sample": names,
                "Score_Distance": self.SD_,
                "Orthogonal_Distance": self.OD_,
            }
        )
        if hasattr(self, "sd_limit_") and hasattr(self, "od_limit_"):
            df["Exceeds_SD_Limit"] = df["Score_Distance"] > self.sd_limit_
            df["Exceeds_OD_Limit"] = df["Orthogonal_Distance"] > self.od_limit_
        if y_true is not None:
            df["True_Class"] = np.asarray(y_true)
        return df

    # ------------------------------------------------------------------
    # Internal fitting and prediction
    # ------------------------------------------------------------------
    def _fit_numeric(self, X, y_numeric, x_mean=None, x_std=None, y_mean=None, y_std=None):
        X = np.asarray(X, dtype=float)
        y_numeric = np.asarray(y_numeric, dtype=float)
        n_samples, n_features = X.shape

        self.x_mean_ = np.nanmean(X, axis=0) if x_mean is None else np.asarray(x_mean, dtype=float)
        self.x_std_ = self._scale_vector(X) if x_std is None else np.asarray(x_std, dtype=float)
        self.x_std_ = self._sanitize_scale(self.x_std_)
        E = (X - self.x_mean_) / self.x_std_
        E_orig = E.copy()

        self.y_mean_ = float(np.nanmean(y_numeric)) if y_mean is None else float(y_mean)
        self.y_std_ = float(np.nanstd(y_numeric, ddof=1)) if y_std is None else float(y_std)
        if self.y_std_ == 0:
            self.y_std_ = 1.0
        f = (y_numeric - self.y_mean_) / self.y_std_

        ss_x_total = np.nansum(E_orig**2)
        ss_y_total = np.nansum((y_numeric - self.y_mean_) ** 2)
        if ss_y_total == 0:
            raise ValueError("The response has zero variance.")

        n_ortho = int(self.n_ortho_)
        self.T_ortho_ = np.zeros((n_samples, n_ortho))
        self.P_ortho_ = np.zeros((n_features, n_ortho))
        self.W_ortho_ = np.zeros((n_features, n_ortho))
        self.R2Y_abs_ = []
        r2x_ortho_list = []

        w_pred_0, t_pred_0, p_pred_0, c_pred_0 = self._predictive_component(E, f)
        y_pred_0 = (t_pred_0 * c_pred_0) * self.y_std_ + self.y_mean_
        r2y_cum_0 = 1.0 - np.nansum((y_numeric - y_pred_0) ** 2) / ss_y_total
        self.R2Y_abs_.append(r2y_cum_0)

        if n_ortho == 0:
            self.w_pred_ = w_pred_0
            self.t_pred_ = t_pred_0
            self.p_pred_ = p_pred_0
            self.c_pred_ = c_pred_0
            self.fitted_values_ = y_pred_0

        for i in range(n_ortho):
            w, t, p, _ = self._predictive_component(E, f)
            w_ortho = p - np.dot(w.T, p) * w
            w_ortho /= np.linalg.norm(w_ortho)

            valid_mask = ~np.isnan(E)
            t_ortho = np.nansum(E * w_ortho, axis=1)
            t_ortho /= np.sum(valid_mask * (w_ortho**2), axis=1)

            p_ortho = np.nansum(E * t_ortho[:, np.newaxis], axis=0)
            p_ortho /= np.nansum(t_ortho**2)

            self.T_ortho_[:, i] = t_ortho
            self.P_ortho_[:, i] = p_ortho
            self.W_ortho_[:, i] = w_ortho
            r2x_ortho_list.append(
                np.nansum(np.outer(t_ortho, p_ortho) ** 2) / ss_x_total
            )

            E = E - np.outer(t_ortho, p_ortho)
            w_pred, t_pred, p_pred, c_pred = self._predictive_component(E, f)

            y_pred = (t_pred * c_pred) * self.y_std_ + self.y_mean_
            r2y_cum = 1.0 - np.nansum((y_numeric - y_pred) ** 2) / ss_y_total
            self.R2Y_abs_.append(r2y_cum)

            if i == n_ortho - 1:
                self.w_pred_ = w_pred
                self.t_pred_ = t_pred
                self.p_pred_ = p_pred
                self.c_pred_ = c_pred
                self.fitted_values_ = y_pred

        r2x_pred = np.nansum(np.outer(self.t_pred_, self.p_pred_) ** 2) / ss_x_total
        self.R2X_comp_ = [r2x_pred] + r2x_ortho_list
        self.R2X_cum_list_ = list(np.cumsum(self.R2X_comp_))
        self.R2Y_comp_ = []
        self.R2Y_cum_list_ = []
        for i, r2y_abs in enumerate(self.R2Y_abs_):
            if i == 0:
                self.R2Y_comp_.append(r2y_abs)
                self.R2Y_cum_list_.append(r2y_abs)
            else:
                self.R2Y_comp_.append(r2y_abs - self.R2Y_abs_[i - 1])
                self.R2Y_cum_list_.append(r2y_abs - self.R2Y_abs_[0])

        self.R2Y_ = self.R2Y_abs_[-1]
        self.r2y_ = self.R2Y_
        self.RMSEE_ = self._rmsee(y_numeric, self.fitted_values_)
        self.rmsee_ = self.RMSEE_
        self.fitted_class_ = self.predict_from_numeric(self.fitted_values_)
        self._compute_vip(n_features)
        self._compute_feature_correlations(X)
        self._compute_outlier_diagnostics(E_orig)
        self.coef_ = (self.w_pred_ * self.c_pred_ * self.y_std_) / self.x_std_
        self.intercept_ = self.y_mean_ - np.dot(self.x_mean_, self.coef_)
        return self

    def _predictive_component(self, E, f):
        w = np.nansum(E * f[:, np.newaxis], axis=0)
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            raise ValueError("Could not extract a predictive component.")
        w /= norm_w

        valid_mask = ~np.isnan(E)
        t = np.nansum(E * w, axis=1)
        t /= np.sum(valid_mask * (w**2), axis=1)

        p = np.nansum(E * t[:, np.newaxis], axis=0)
        p /= np.nansum(t**2)
        c = np.nansum(f * t) / np.nansum(t**2)
        return w, t, p, c

    def _predict_continuous(self, X):
        check_is_fitted(self, "w_pred_")
        x_prepared = self._prepare_X(X, reset=False)
        X_arr = self._apply_zero_variance_filter(x_prepared.values, fit=False)
        return self._predict_continuous_prepared(X_arr)

    def _predict_continuous_prepared(self, X):
        scores = self._project_scores_prepared(X)
        return (scores["t_pred"] * self.c_pred_) * self.y_std_ + self.y_mean_

    def _project_scores(self, X):
        check_is_fitted(self, "w_pred_")
        x_prepared = self._prepare_X(X, reset=False)
        X_arr = self._apply_zero_variance_filter(x_prepared.values, fit=False)
        return self._project_scores_prepared(X_arr)

    def _project_scores_prepared(self, X):
        E = (np.asarray(X, dtype=float) - self.x_mean_) / self.x_std_
        t_ortho_cols = []
        for i in range(self.n_ortho_):
            valid_mask = ~np.isnan(E)
            w_o_sq = self.W_ortho_[:, i] ** 2
            t_ortho = np.nansum(E * self.W_ortho_[:, i], axis=1)
            t_ortho /= np.sum(valid_mask * w_o_sq, axis=1)
            E -= np.outer(t_ortho, self.P_ortho_[:, i])
            t_ortho_cols.append(t_ortho)

        valid_mask = ~np.isnan(E)
        w_p_sq = self.w_pred_**2
        t_pred = np.nansum(E * self.w_pred_, axis=1)
        t_pred /= np.sum(valid_mask * w_p_sq, axis=1)

        if t_ortho_cols:
            t_ortho = np.column_stack(t_ortho_cols)
        else:
            t_ortho = np.zeros((X.shape[0], 0))
        return {"t_pred": t_pred, "t_ortho": t_ortho}

    def predict_from_numeric(self, y_pred_num):
        if len(self.classes_) == 2:
            pred_idx = (np.asarray(y_pred_num) > 0.5).astype(int)
        else:
            pred_idx = np.clip(
                np.round(y_pred_num).astype(int), 0, len(self.classes_) - 1
            )
        return self.label_encoder.inverse_transform(pred_idx)

    # ------------------------------------------------------------------
    # Validation, CV, and compatibility helpers
    # ------------------------------------------------------------------
    def _validate_params(self):
        if self.compatibility not in {"sklearn", "ropls"}:
            raise ValueError("compatibility must be 'sklearn' or 'ropls'.")
        if self.cv_preprocess not in {"fold", "global"}:
            raise ValueError("cv_preprocess must be 'fold' or 'global'.")
        if self.scale not in {"standard", "center", "pareto", "none"}:
            raise ValueError("scale must be 'standard', 'center', 'pareto', or 'none'.")
        if self.missing not in {"error", "allow"}:
            raise ValueError("missing must be 'error' or 'allow'.")
        if self.selection_rule not in {"auto", "sklearn_q2", "ropls"}:
            raise ValueError("selection_rule must be 'auto', 'sklearn_q2', or 'ropls'.")

    def _prepare_X(self, X, reset):
        feature_names = X.columns.astype(str).tolist() if hasattr(X, "columns") else None
        sample_names = X.index.astype(str).tolist() if hasattr(X, "index") else None
        force_all_finite = "allow-nan" if self.missing == "allow" else True
        try:
            X_arr = check_array(
                X,
                dtype=float,
                ensure_2d=True,
                ensure_all_finite=force_all_finite,
            )
        except TypeError:
            X_arr = check_array(
                X,
                dtype=float,
                ensure_2d=True,
                force_all_finite=force_all_finite,
            )
        if not reset and hasattr(self, "n_features_in_"):
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X_arr.shape[1]} features, but this estimator was "
                    f"fitted with {self.n_features_in_} features."
                )
        return _PreparedX(X_arr, feature_names, sample_names)

    def _apply_zero_variance_filter(self, X, fit):
        if fit:
            variances = np.nanvar(X, axis=0, ddof=1)
            if self.remove_zero_variance:
                mask = variances > np.finfo(float).eps
            else:
                mask = np.ones(X.shape[1], dtype=bool)
            if not np.any(mask):
                raise ValueError("All predictor variables have near-zero variance.")
            self.zero_variance_mask_ = mask
            self.xZeroVarVi_ = np.where(~mask)[0]
            return X[:, mask]
        return X[:, self.zero_variance_mask_]

    def _resolve_n_ortho(self, X, y_num):
        if self.n_ortho in (None, "auto"):
            return self._find_best_n_ortho(X, y_num)
        n_ortho = int(self.n_ortho)
        if n_ortho < 0:
            raise ValueError("n_ortho must be non-negative, 'auto', or None.")
        if n_ortho + 1 > min(X.shape):
            raise ValueError("n_ortho + 1 cannot exceed min(n_samples, n_features).")
        return n_ortho

    def _find_best_n_ortho(self, X, y_num):
        rule = self._effective_selection_rule()
        best_n = 0
        prev_q2 = None

        for n in range(0, int(self.max_ortho) + 1):
            temp = self._new_like(n_ortho=n)
            temp.label_encoder = self.label_encoder
            temp.classes_ = self.classes_
            temp._is_categorical = True
            temp.feature_names_in_ = self.kept_feature_names_
            temp.sample_names_in_ = None
            temp.n_features_in_ = X.shape[1]
            temp.zero_variance_mask_ = np.ones(X.shape[1], dtype=bool)
            temp.n_model_features_in_ = X.shape[1]
            temp.kept_feature_names_ = self.kept_feature_names_
            temp.n_ortho_ = n
            temp._fit_numeric(X, y_num)
            temp.compute_q2(X, self.label_encoder.inverse_transform(y_num.astype(int)))

            if n == 0:
                best_n = 0
                prev_q2 = temp.Q2_
                continue

            if rule == "ropls":
                keep = temp.R2Y_comp_[-1] >= 0.01 and temp.Q2_comp_[-1] >= 0.01
            else:
                keep = (temp.Q2_ - prev_q2) >= self.auto_q2_min_delta

            if keep:
                best_n = n
                prev_q2 = temp.Q2_
            else:
                break
        return best_n

    def _iter_cv_splits(self, X, y_num) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        strategy = self._effective_cv_strategy_name()
        n = X.shape[0]
        if self.cv_folds < 2 or self.cv_folds > n:
            raise ValueError("cv_folds must be between 2 and the number of samples.")

        if not isinstance(self.cv_strategy, str) and hasattr(self.cv_strategy, "split"):
            yield from self.cv_strategy.split(X, y_num)
        elif strategy == "ropls_venetian":
            idx = np.arange(n)
            for i in range(self.cv_folds):
                test = idx[i:: self.cv_folds]
                train = np.setdiff1d(idx, test)
                yield train, test
        elif strategy == "stratified_venetian":
            folds = [([], []) for _ in range(self.cv_folds)]
            for class_val in np.unique(y_num):
                idx_c = np.where(y_num == class_val)[0]
                for i in range(self.cv_folds):
                    test_c = idx_c[i:: self.cv_folds]
                    train_c = np.setdiff1d(idx_c, test_c)
                    folds[i][0].extend(train_c)
                    folds[i][1].extend(test_c)
            for train, test in folds:
                yield np.array(train, dtype=int), np.array(test, dtype=int)
        elif strategy == "kfold":
            splitter = KFold(
                n_splits=self.cv_folds,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            yield from splitter.split(X, y_num)
        else:
            splitter = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            yield from splitter.split(X, y_num)

    def _effective_cv_strategy_name(self):
        if not isinstance(self.cv_strategy, str) and hasattr(self.cv_strategy, "split"):
            return self.cv_strategy.__class__.__name__
        if self.cv_strategy == "auto":
            return "ropls_venetian" if self.compatibility == "ropls" else "stratified"
        return self.cv_strategy

    def _effective_cv_preprocess(self):
        if self.compatibility == "ropls" and self.cv_preprocess == "fold":
            return "global"
        return self.cv_preprocess

    def _effective_selection_rule(self):
        if self.selection_rule == "auto":
            return "ropls" if self.compatibility == "ropls" else "sklearn_q2"
        return self.selection_rule

    def _effective_pvalue_method(self):
        if self.pvalue_method == "auto":
            return "ropls" if self.compatibility == "ropls" else "standard"
        return self.pvalue_method

    def _effective_rmsee_correction(self):
        if self.rmsee_correction == "auto":
            return "ropls" if self.compatibility == "ropls" else "sklearn"
        return self.rmsee_correction

    def _new_like(self, *, n_ortho):
        return OPLSDA(
            n_ortho=n_ortho,
            cv_folds=self.cv_folds,
            max_ortho=self.max_ortho,
            n_perms=self.n_perms,
            n_jobs=1,
            vip_method=self.vip_method,
            compatibility=self.compatibility,
            cv_strategy=self.cv_strategy,
            cv_preprocess=self.cv_preprocess,
            scale=self.scale,
            selection_rule=self.selection_rule,
            auto_q2_min_delta=self.auto_q2_min_delta,
            remove_zero_variance=False,
            missing=self.missing,
            shuffle=self.shuffle,
            random_state=self.random_state,
            pvalue_method=self.pvalue_method,
            rmsee_correction=self.rmsee_correction,
            allow_multiclass=self.allow_multiclass,
        )

    def _single_permutation(self, X, y_numeric, seed):
        rng = check_random_state(seed)
        y_perm = rng.permutation(y_numeric)
        model_perm = self._new_like(n_ortho=self.n_ortho_)
        model_perm.label_encoder = self.label_encoder
        model_perm.classes_ = self.classes_
        model_perm._is_categorical = True
        model_perm.feature_names_in_ = self.kept_feature_names_
        model_perm.sample_names_in_ = None
        model_perm.n_features_in_ = X.shape[1]
        model_perm.zero_variance_mask_ = np.ones(X.shape[1], dtype=bool)
        model_perm.n_model_features_in_ = X.shape[1]
        model_perm.kept_feature_names_ = self.kept_feature_names_
        model_perm.n_ortho_ = self.n_ortho_
        model_perm._fit_numeric(X, y_perm)
        q2 = model_perm.compute_q2(
            X, self.label_encoder.inverse_transform(y_perm.astype(int))
        )
        return model_perm.R2Y_abs_[-1], q2

    # ------------------------------------------------------------------
    # Numeric utilities
    # ------------------------------------------------------------------
    def _scale_vector(self, X):
        if self.scale == "none":
            self.x_mean_ = np.zeros(X.shape[1], dtype=float)
            return np.ones(X.shape[1], dtype=float)
        if self.scale == "center":
            return np.ones(X.shape[1], dtype=float)
        sd = np.nanstd(X, axis=0, ddof=1)
        if self.scale == "pareto":
            return np.sqrt(sd)
        return sd

    @staticmethod
    def _sanitize_scale(scale):
        scale = np.asarray(scale, dtype=float)
        scale[(scale == 0) | ~np.isfinite(scale)] = 1.0
        return scale

    def _rmsee(self, y_true, y_pred):
        rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))
        if self._effective_rmsee_correction() == "ropls":
            denom = len(y_true) - (1 + 1 + self.n_ortho_)
            if denom > 0:
                return rmse * np.sqrt(len(y_true) / denom)
        return rmse

    def _compute_vip(self, n_features):
        if self.vip_method == "vip4":
            sxp = np.nansum(np.outer(self.t_pred_, self.p_pred_) ** 2)
            sxo = sum(
                np.nansum(np.outer(self.T_ortho_[:, j], self.P_ortho_[:, j]) ** 2)
                for j in range(self.n_ortho_)
            )
            ssx_cum = sxp + sxo
            syp = np.nansum((self.t_pred_ * self.c_pred_) ** 2)
            ssy_cum = syp
            kp = n_features / ((sxp / ssx_cum) + (syp / ssy_cum))
            p_norm = self.p_pred_ / np.sqrt(np.nansum(self.p_pred_**2))
            term_x = (p_norm**2 * sxp) / ssx_cum
            term_y = (p_norm**2 * syp) / ssy_cum
            self.vip_ropls_ = np.sqrt(kp * (term_x + term_y))
        else:
            if self.n_ortho_ > 0:
                W_all = np.column_stack((self.w_pred_, self.W_ortho_))
                P_all = np.column_stack((self.p_pred_, self.P_ortho_))
                W_star = np.dot(W_all, np.linalg.inv(np.dot(P_all.T, W_all)))
                w_star_pred = W_star[:, 0]
            else:
                w_star_pred = self.w_pred_.copy()
            w_star_pred /= np.linalg.norm(w_star_pred)
            self.vip_ropls_ = np.sqrt(n_features * (w_star_pred**2))
        self.vip_ = self.vip_ropls_

    def _compute_feature_correlations(self, X):
        covariances = np.zeros(X.shape[1])
        correlations = np.zeros(X.shape[1])
        X_centered = X - np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mask = ~np.isnan(X_centered[:, j])
            t_valid = self.t_pred_[mask]
            x_valid = X_centered[mask, j]
            if len(t_valid) > 1:
                covariances[j] = np.cov(t_valid, x_valid)[0, 1]
                std_t = np.std(t_valid, ddof=1)
                std_x = np.std(x_valid, ddof=1)
                if std_t > 0 and std_x > 0:
                    correlations[j] = covariances[j] / (std_t * std_x)
        self.covariances_ = covariances
        self.correlations_ = correlations

    def _compute_outlier_diagnostics(self, E_orig):
        E_res = E_orig - np.outer(self.t_pred_, self.p_pred_)
        for i in range(self.n_ortho_):
            E_res -= np.outer(self.T_ortho_[:, i], self.P_ortho_[:, i])
        self.OD_ = np.sqrt(np.nanmean(E_res**2, axis=1) * E_res.shape[1])

        T_all = (
            np.column_stack((self.t_pred_, self.T_ortho_))
            if self.n_ortho_ > 0
            else self.t_pred_[:, np.newaxis]
        )
        T_var = np.var(T_all, axis=0, ddof=1)
        T_var[T_var == 0] = 1e-10
        self.SD_ = np.sqrt(np.sum((T_all**2) / T_var, axis=1))
        self.sd_limit_ = np.sqrt(chi2.ppf(0.95, df=1 + self.n_ortho_))

        if len(self.OD_) > 1:
            od_23 = self.OD_ ** (2 / 3)
            self.od_limit_ = (np.mean(od_23) + norm.ppf(0.95) * np.std(od_23, ddof=1)) ** (3 / 2)
        else:
            self.od_limit_ = np.max(self.OD_) * 1.1

    def _kept_feature_names(self):
        if self.feature_names_in_ is None:
            return [f"F_{i}" for i in range(self.n_model_features_in_)]
        return [
            name
            for name, keep in zip(self.feature_names_in_, self.zero_variance_mask_)
            if keep
        ]

    def _resolve_feature_names(self, feature_names):
        if feature_names is None:
            return list(self.kept_feature_names_)
        names = list(feature_names)
        if len(names) == self.n_features_in_:
            return [name for name, keep in zip(names, self.zero_variance_mask_) if keep]
        if len(names) == self.n_model_features_in_:
            return names
        raise ValueError("feature_names length does not match fitted feature count.")

    def _resolve_sample_names(self, sample_names):
        if sample_names is not None:
            return list(sample_names)
        if self.sample_names_in_ is not None:
            return list(self.sample_names_in_)
        return [f"S_{i}" for i in range(self.t_pred_.shape[0])]


OPLSDAClassifier = OPLSDA
