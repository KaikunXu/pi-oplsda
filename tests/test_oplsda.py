# tests/test_oplsda.py
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from piopls import OPLSDA, OPLSDA_Visualizer


@pytest.fixture
def dummy_data():
    """
    Generate virtual metabolomics data for testing.
    Contains 20 samples (two groups) and 15 features.
    """
    np.random.seed(42)
    n_samples = 20
    n_features = 15
    
    # Randomly generate base data
    X = np.random.rand(n_samples, n_features)
    
    # Change [0, 1] to real labels
    y = np.array(['Group_A'] * 10 + ['Group_B'] * 10)
    
    # Artificially create differential features (Biomarkers)
    X[:10, 0] += 2.0
    X[10:, 1] += 2.0
    
    # Wrap as Pandas DataFrame and Series
    feature_names = [f"Metabolite_{i}" for i in range(n_features)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    
    df_X = pd.DataFrame(X, columns=feature_names, index=sample_names)
    df_y = pd.Series(y, index=sample_names, name="Group")
    
    return df_X, df_y


def test_model_fit_and_basic_metrics(dummy_data):
    """Test basic model fitting and R2X/R2Y metric calculation."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    # Updated to match current cumulative list attributes
    assert hasattr(model, 'R2Y_abs_'), "Model misses R2Y_abs_ after fitting."
    assert hasattr(model, 'R2X_comp_'), "Model misses R2X_comp_ after fitting."
    assert model.R2Y_abs_[-1] > 0, "Cumulative R2Y should be greater than 0."


def test_compute_q2_and_summary_df(dummy_data):
    """Test cross-validation Q2 calculation and summary dataframe output."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    # Calling summary_df before computing Q2 should raise an error
    with pytest.raises(ValueError, match="Model must compute Q2 first"):
        model.get_summary_df()
        
    q2 = model.compute_q2(df_X, df_y)
    assert isinstance(q2, float), "Q2 should return a float value."
    
    df_summary = model.get_summary_df()
    assert isinstance(df_summary, pd.DataFrame)
    assert 'Q2' in df_summary.columns
    assert 'R2X(cum)' in df_summary.columns
    assert len(df_summary) == 2, "n_ortho=1 should yield 2 rows (p1 and o1)."


def test_get_model_info_df(dummy_data):
    """Test the generation of the global model info dataframe."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    model.compute_q2(df_X, df_y)
    
    # Run permutation test to ensure p-values are attached
    model.permutation_test(df_X, df_y, n_perms=5, n_jobs=1)
    
    df_info = model.get_model_info_df()
    assert isinstance(df_info, pd.DataFrame)
    assert len(df_info) == 1, "Model info should be a single-row DataFrame."
    assert 'RMSEE' in df_info.columns
    assert 'pR2Y' in df_info.columns
    assert 'pQ2' in df_info.columns


def test_features_df_and_vip_ranking(dummy_data):
    """Test feature output table and VIP ranking extraction."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    df_features = model.get_features_df()
    
    assert 'VIP' in df_features.columns
    assert 'Feature' in df_features.columns
    assert 'Covariance (p1)' in df_features.columns
    assert 'Correlation (pcorr1)' in df_features.columns
    assert 'Loading_Weight' in df_features.columns
    
    f_name = df_features['Feature'].iloc[0]
    assert f_name.startswith("Metabolite_"), "Failed to extract original names."
    
    vips = df_features['VIP'].values
    is_sorted = all(vips[i] >= vips[i+1] for i in range(len(vips)-1))
    assert is_sorted, "Features DataFrame is not sorted by VIP descending."
    
    top_feature = df_features['Feature'].iloc[0]
    valid_tops = ["Metabolite_0", "Metabolite_1"]
    assert top_feature in valid_tops, "Biomarker is not at the top."
    assert df_features['VIP'].iloc[0] > 1.0, "Top VIP should be > 1.0."


def test_scores_df_metadata(dummy_data):
    """Test sample score table and fitted values."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    df_scores = model.get_scores_df(y_true=df_y.values)
    
    assert 'Sample' in df_scores.columns
    # Updated column names to match the modern OPLSDA plotting API
    assert 't_pred (p1)' in df_scores.columns
    assert 't_ortho (o1)' in df_scores.columns
    
    assert 'True_Class' in df_scores.columns
    assert 'Fitted_Value' in df_scores.columns
    assert 'Fitted_Class' in df_scores.columns
    assert 'Match_Status' in df_scores.columns 
    
    assert df_scores['Sample'].iloc[0] == "Sample_0", "Name extraction failed."


def test_permutation_test(dummy_data):
    """Test multi-process permutation test function."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y) 
    
    perm_res = model.permutation_test(df_X, df_y, n_perms=5, n_jobs=1)
    
    assert 'p_R2Y' in perm_res
    # Updated to match current dictionary key
    assert 'p_Q2Y' in perm_res
    assert len(perm_res['perm_R2Y']) == 5, "Length of permutations mismatch."


def test_sacurine_example_data():
    """
    Test the entire workflow using the built-in Sacurine dataset.
    Writes generated plots to a project-local ignored directory. This avoids
    pytest's default system temp directory and Windows cleanup permission issues.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "src", "piopls", "data")
    
    path_X = os.path.join(data_dir, "sacurine_X.csv")
    path_Y = os.path.join(data_dir, "sacurine_Y.csv")
    
    if not os.path.exists(path_X) or not os.path.exists(path_Y):
        pytest.skip(f"Data not found at {data_dir}. Skipping sacurine test.")
        
    df_X = pd.read_csv(path_X, index_col=0)
    df_Y = pd.read_csv(path_Y, index_col=0)
    y_labels = df_Y.iloc[:, 0].values
    
    assert df_X.shape[0] > 0 and df_X.shape[1] > 0, "Loaded X matrix is empty."
    
    model = OPLSDA(cv_folds=7)
    model.fit_pipeline(df_X, y_labels)
    
    df_info = model.get_model_info_df()
    assert not df_info.empty, "Model info DataFrame is empty."
    
    df_summary = model.get_summary_df()
    assert not df_summary.empty, "Summary DataFrame is empty."
    
    perm_results = model.permutation_test(
        df_X, y_labels, n_perms=10, n_jobs=-1
    )
    # Updated keys to align with current API
    assert 'p_R2Y' in perm_results and 'p_Q2Y' in perm_results
    
    vis = OPLSDA_Visualizer(
        model=model,
        y=y_labels, 
        vip_threshold=1.0,
        top_n_vip=15
    )
    
    artifact_dir = Path(project_root) / ".pytest-artifacts" / "test_sacurine_example_data"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_path = artifact_dir / "test_oplsda_diagnostic_plots.png"
    vis.plot_all(perm_results=perm_results, save_path=save_path)
    
    assert os.path.exists(save_path), "Failed to generate diagnostic plots."
    
    df_features = model.get_features_df()
    assert len(df_features) == df_X.shape[1], "Features length mismatch."
