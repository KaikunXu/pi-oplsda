# tests/test_ropls_benchmark.py
import os
import sys
import subprocess
import logging
import pytest
import numpy as np
import pandas as pd

import piopls
from piopls import OPLSDA, load_sacurine


def test_ropls_benchmark():
    """Validates pi-oplsda metrics against R ropls package via rpy2."""

    # =========================================================================
    # 1. Bulletproof R Environment Initialization (Targeting R-4.5.3)
    # =========================================================================
    valid_r_home = None

    # Prioritize the known hardcoded path on this system
    primary_target = "D:/R/R-4.5.3"
    candidates = [primary_target]

    # Candidate 1: System command line search
    try:
        out = subprocess.check_output(
            ["R", "RHOME"], text=True, stderr=subprocess.DEVNULL
        )
        candidates.append(out.strip())
    except Exception:
        pass

    # Candidate 2: Windows Registry search
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R"
            ) as key:
                candidates.append(winreg.QueryValueEx(key, "InstallPath")[0])
        except Exception:
            pass

    # Validate if candidate paths actually contain R executable
    for c in candidates:
        if c and os.path.exists(c):
            r_exe1 = os.path.join(c, "bin", "R.exe")
            r_exe2 = os.path.join(c, "bin", "x64", "R.exe")
            if os.path.exists(r_exe1) or os.path.exists(r_exe2):
                valid_r_home = c
                break

    if not valid_r_home:
        pytest.skip("Valid R installation not found. Skipping benchmark.")

    # Inject R paths into system environment variables
    os.environ["R_HOME"] = valid_r_home
    r_bin_path = os.path.join(valid_r_home, "bin", "x64")
    if not os.path.exists(r_bin_path):
        r_bin_path = os.path.join(valid_r_home, "bin")

    os.environ["PATH"] = r_bin_path + os.pathsep + os.environ.get("PATH", "")
    os.environ["LANGUAGE"] = "en"
    os.environ["LC_ALL"] = "C"

    # =========================================================================
    # 2. Safe rpy2 Import and Execution
    # =========================================================================
    try:
        from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
        rpy2_logger.setLevel(logging.ERROR)

        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import PackageNotInstalledError
    except Exception as e:
        pytest.skip(f"rpy2 backend failed to initialize: {e}")

    print("\n" + "=" * 60)
    print(f"   Benchmarking pi-oplsda vs ropls (R Env: {valid_r_home})")
    print("=" * 60)

    print("\n[1/5] Loading sacurine dataset...")
    X, y, _, _ = load_sacurine()

    print("[2/5] Fitting Python pi-oplsda model...")
    model_py = OPLSDA(n_ortho=1, cv_folds=7)
    model_py.fit_pipeline(X, y)

    r2y_py = model_py.R2Y_abs_[-1]
    q2_py = model_py.Q2_abs_[-1]
    vip_py = model_py.vip_ropls_

    print("[3/5] Fitting R ropls model via rpy2...")
    try:
        ropls = importr("ropls")
    except PackageNotInstalledError:
        pytest.skip("R package 'ropls' is missing. Skipping benchmark.")
    except Exception as e:
        pytest.skip(f"Unexpected error loading ropls: {e}")

    rpy_conv = ro.default_converter + pandas2ri.converter + numpy2ri.converter

    # Phase A: Pass data to R and fit model inside the converter context
    with localconverter(rpy_conv):
        # Explicitly convert numpy array to R matrix
        r_matrix = ro.conversion.get_conversion().py2rpy(X)
        r_group = ro.FactorVector(ro.StrVector(y))

        ro.r("sink('NUL')")
        try:
            model_r = ropls.opls(
                r_matrix, r_group, predI=1, orthoI=1, 
                crossvalI=7, permI=0, fig_pdfC="none"
            )
        finally:
            ro.r("sink()")

    # Phase B: Extract S4 slots in pure R environment (outside context)
    print("[4/5] Extracting summary slots from R S4 object...")
    ro.r('''
    get_metrics <- function(model) {
        list(sum_df=model@summaryDF, vip=model@vipVn)
    }
    ''')
    get_metrics = ro.globalenv['get_metrics']
    
    # Returns a pure R list, fully supporting the .rx2 extraction method
    res_r = get_metrics(model_r)

    # Phase C: Convert back to Python explicitly using specific converters
    # 1. Convert Pandas DataFrame
    pd_conv = ro.default_converter + pandas2ri.converter
    with localconverter(pd_conv):
        sum_df_r = ro.conversion.get_conversion().rpy2py(res_r.rx2('sum_df'))
        # Re-initialize to ensure standard pandas DataFrame formatting
        sum_df_r = pd.DataFrame(sum_df_r)

    # 2. Convert Numpy Array
    np_conv = ro.default_converter + numpy2ri.converter
    with localconverter(np_conv):
        vip_r = ro.conversion.get_conversion().rpy2py(res_r.rx2('vip'))
        vip_r = np.array(vip_r)

    r2y_r = sum_df_r['R2Y(cum)'].values[-1]
    q2_r = sum_df_r['Q2(cum)'].values[-1]

    print("[5/5] Validating numerical equivalence...")
    tol = 1e-2

    diff_r2y = abs(r2y_py - r2y_r)
    print(f"      - R2Y (Python): {r2y_py:.6f} | R2Y (R): {r2y_r:.6f}")
    assert diff_r2y < tol, "R2Y mismatch between Python and R!"

    diff_q2 = abs(q2_py - q2_r)
    print(f"      - Q2  (Python): {q2_py:.6f} | Q2  (R): {q2_r:.6f}")
    assert diff_q2 < tol, "Q2 mismatch between Python and R!"

    vip_match = np.allclose(vip_py, vip_r, atol=tol)
    print(f"      - VIP Arrays Match: {vip_match}")
    assert vip_match, "VIP mismatch between Python and R!"

    print("\n" + "=" * 60)
    print("   SUCCESS: pi-oplsda is mathematically equivalent to ropls!")
    print("=" * 60)