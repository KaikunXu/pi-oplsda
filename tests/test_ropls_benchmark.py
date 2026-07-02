# tests/test_ropls_benchmark.py
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from piopls import OPLSDA, load_sacurine


def _candidate_r_homes():
    candidates = []

    env_home = os.environ.get("PI_OPLSDA_R_HOME")
    if env_home:
        candidates.append(Path(env_home))

    # Prefer the standalone R installation over any conda-provided R.
    candidates.append(Path("D:/R/R-4.5.2"))

    if os.environ.get("R_HOME"):
        candidates.append(Path(os.environ["R_HOME"]))

    try:
        out = subprocess.check_output(
            ["R", "RHOME"],
            text=True,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
        ).strip()
        if out:
            candidates.append(Path(out))
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R"
            ) as key:
                candidates.append(Path(winreg.QueryValueEx(key, "InstallPath")[0]))
        except Exception:
            pass

    for root in (Path("D:/R"), Path("C:/Program Files/R")):
        if root.exists():
            candidates.extend(sorted(root.glob("R-*"), reverse=True))

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        yield resolved


def _configure_r_runtime():
    selected = None
    selected_rscript = None

    for candidate in _candidate_r_homes():
        r_exe = candidate / "bin" / "x64" / "R.exe"
        rscript = candidate / "bin" / "x64" / "Rscript.exe"
        stats_dll = candidate / "library" / "stats" / "libs" / "x64" / "stats.dll"

        if not r_exe.exists():
            r_exe = candidate / "bin" / "R.exe"
        if not rscript.exists():
            rscript = candidate / "bin" / "Rscript.exe"
        if stats_dll.exists() and r_exe.exists() and rscript.exists():
            selected = candidate
            selected_rscript = rscript
            break

    if selected is None or selected_rscript is None:
        pytest.skip("Valid standalone R installation not found.")

    bin_dirs = [selected / "bin" / "x64", selected / "bin"]
    existing_bin_dirs = [str(path) for path in bin_dirs if path.exists()]

    os.environ["R_HOME"] = str(selected)
    os.environ["PATH"] = os.pathsep.join(existing_bin_dirs + [os.environ.get("PATH", "")])
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

    # LC_ALL=C breaks embedded R package loading on this Windows R build.
    os.environ.pop("LC_ALL", None)
    os.environ.pop("LANGUAGE", None)

    dll_handles = []
    if hasattr(os, "add_dll_directory"):
        for directory in existing_bin_dirs:
            dll_handles.append(os.add_dll_directory(directory))

    probe = subprocess.run(
        [str(selected_rscript), "-e", "library(stats); cat('stats OK\\n')"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip(f"R found but base package stats failed to load: {probe.stderr}")

    return str(selected), dll_handles


def _patch_rpy2_windows_dll_lookup(r_home):
    if os.name != "nt":
        return

    import rpy2.situation as rpy2_situation

    original_get_r_flags = rpy2_situation.get_r_flags
    if getattr(original_get_r_flags, "_piopls_windows_patch", False):
        return

    def get_r_flags_patched(r_home_arg, flags):
        if flags == "--ldflags":
            base = Path(r_home_arg or r_home)
            lib_dirs = [
                str(path) for path in (base / "bin" / "x64", base / "bin") if path.exists()
            ]
            if lib_dirs:
                return SimpleNamespace(I=None, L=lib_dirs, l=[]), []

        return original_get_r_flags(r_home_arg, flags)

    get_r_flags_patched._piopls_windows_patch = True
    rpy2_situation.get_r_flags = get_r_flags_patched


def _abs_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    assert a.shape == b.shape
    return abs(float(np.corrcoef(a, b)[0, 1]))


def test_ropls_benchmark_auto_ortho():
    """Validate pi-oplsda against R ropls with automatic orthogonal selection."""

    if "rpy2.robjects" in sys.modules or "rpy2.rinterface" in sys.modules:
        pytest.skip("rpy2 was imported before the R runtime could be configured.")

    r_home, dll_handles = _configure_r_runtime()
    _patch_rpy2_windows_dll_lookup(r_home)

    try:
        from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

        rpy2_logger.setLevel(logging.ERROR)
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import PackageNotInstalledError, importr
    except Exception as exc:
        pytest.skip(f"rpy2 backend failed to initialize: {exc}")

    try:
        ropls = importr("ropls")
    except PackageNotInstalledError:
        pytest.skip("R package 'ropls' is missing from the standalone R installation.")

    X, y, _, _ = load_sacurine()

    model_py = OPLSDA(
        n_ortho="auto",
        max_ortho=10,
        cv_folds=7,
        compatibility="ropls",
        random_state=42,
        n_perms=100,
        n_jobs=-1,
    )
    model_py.fit(X, y)
    model_py.compute_q2(X, y)

    rpy_conv = ro.default_converter + pandas2ri.converter + numpy2ri.converter
    with localconverter(rpy_conv):
        r_matrix = ro.conversion.get_conversion().py2rpy(X)
        r_group = ro.FactorVector(ro.StrVector([str(value) for value in y]))
        sink_target = "NUL" if os.name == "nt" else "/dev/null"
        ro.r(f"sink('{sink_target}')")
        try:
            model_r = ropls.opls(
                r_matrix,
                r_group,
                predI=1,
                orthoI=ro.NA_Integer,
                crossvalI=7,
                permI=0,
                fig_pdfC="none",
                info_txtC="none",
            )
        finally:
            ro.r("sink()")

    ro.r(
        """
        get_ropls_benchmark <- function(model) {
            list(
                summary=model@summaryDF,
                vip=model@vipVn,
                score=model@scoreMN,
                loading=model@loadingMN,
                ortho_score=model@orthoScoreMN
            )
        }
        """
    )
    res_r = ro.globalenv["get_ropls_benchmark"](model_r)

    with localconverter(ro.default_converter + pandas2ri.converter):
        summary_r = pd.DataFrame(ro.conversion.get_conversion().rpy2py(res_r.rx2("summary")))
    with localconverter(ro.default_converter + numpy2ri.converter):
        vip_r = np.asarray(ro.conversion.get_conversion().rpy2py(res_r.rx2("vip"))).ravel()
        score_r = np.asarray(ro.conversion.get_conversion().rpy2py(res_r.rx2("score")))
        loading_r = np.asarray(ro.conversion.get_conversion().rpy2py(res_r.rx2("loading")))
        ortho_score_r = np.asarray(
            ro.conversion.get_conversion().rpy2py(res_r.rx2("ortho_score"))
        )

    info_py = model_py.get_model_info_df()
    assert model_py.n_ortho_ == 2
    assert int(summary_r["ort"].iloc[-1]) == 2

    scalar_tol = 2e-3
    for py_col, r_col in [
        ("R2X(cum)", "R2X(cum)"),
        ("R2Y(cum)", "R2Y(cum)"),
        ("Q2(cum)", "Q2(cum)"),
        ("RMSEE", "RMSEE"),
    ]:
        assert abs(float(info_py.loc[0, py_col]) - float(summary_r[r_col].iloc[-1])) < scalar_tol

    assert _abs_corr(model_py.vip_ropls_, vip_r) > 0.9999
    assert _abs_corr(model_py.t_pred_, score_r[:, 0]) > 0.9999
    assert _abs_corr(model_py.p_pred_, loading_r[:, 0]) > 0.9999
    assert _abs_corr(model_py.T_ortho_[:, 0], ortho_score_r[:, 0]) > 0.9999
    assert _abs_corr(model_py.T_ortho_[:, 1], ortho_score_r[:, 1]) > 0.9999

    # Keep DLL directory handles alive until after rpy2/R finalizes this test.
    assert dll_handles is not None
