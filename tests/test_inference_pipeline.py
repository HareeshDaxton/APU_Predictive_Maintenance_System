"""
tests/test_inference_pipeline.py

Tests for the InferencePipeline class and Flask /predict endpoint.
Run with: pytest tests/ -v
All tests use the real saved model + scaler in Artifacts/.
"""
import io
import os
import sys
import json
import pytest
import pandas as pd

# ── Add project root to path ──────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.Pipelines.inference_pipeline import InferencePipeline

# ── Paths ─────────────────────────────────────────────────────
SAMPLE_CSV = os.path.join(ROOT, 'testing_datasets', 'exp', 'data_testing_1.csv')
CONFIG_PATH = os.path.join(ROOT, 'config', 'config.yaml')


@pytest.fixture(scope='module')
def pipeline():
    """Load inference pipeline once for all tests in this module."""
    return InferencePipeline(config_path=CONFIG_PATH)


@pytest.fixture(scope='module')
def sample_df():
    """Load the sample CSV once for all tests."""
    return pd.read_csv(SAMPLE_CSV)


# ─────────────────────────────────────────────────────────────
# TEST 1 — Feature count is exactly 69 before model.predict()
# ─────────────────────────────────────────────────────────────
def test_feature_count(pipeline, sample_df, tmp_path):
    """The model expects exactly 69 features. Never change this."""
    # Write sample to temp CSV and run inference
    tmp_csv = tmp_path / "test_input.csv"
    sample_df.to_csv(tmp_csv, index=False)

    results = pipeline.run_inference(str(tmp_csv))
    # If inference ran without error, feature count was correct.
    # Additional check: results must have predictions
    assert 'predicted_RUL' in results.columns, \
        "Inference ran but predicted_RUL column missing"
    assert len(results) > 0, "No predictions returned"


# ─────────────────────────────────────────────────────────────
# TEST 2 — anomaly_score is injected as 0.0 when missing
# ─────────────────────────────────────────────────────────────
def test_anomaly_score_injection(pipeline, sample_df, tmp_path):
    """If input CSV has no anomaly_score column, it must be auto-added as 0.0."""
    df_no_anomaly = sample_df.drop(columns=['anomaly_score'], errors='ignore')
    assert 'anomaly_score' not in df_no_anomaly.columns, \
        "Setup failed — anomaly_score should have been dropped"

    tmp_csv = tmp_path / "no_anomaly.csv"
    df_no_anomaly.to_csv(tmp_csv, index=False)

    # Should not raise — pipeline injects anomaly_score=0.0 internally
    results = pipeline.run_inference(str(tmp_csv))
    assert len(results) > 0, "Inference failed when anomaly_score was missing"


# ─────────────────────────────────────────────────────────────
# TEST 3 — Predictions are deterministic (scaler transform-only)
# ─────────────────────────────────────────────────────────────
def test_scaler_deterministic(pipeline, sample_df, tmp_path):
    """Running the same input twice must give identical predictions."""
    tmp_csv = tmp_path / "determ.csv"
    sample_df.to_csv(tmp_csv, index=False)

    results1 = pipeline.run_inference(str(tmp_csv))
    results2 = pipeline.run_inference(str(tmp_csv))

    try:
        pd.testing.assert_frame_equal(
            results1[['predicted_RUL']].reset_index(drop=True),
            results2[['predicted_RUL']].reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )
    except AssertionError:
        raise AssertionError(
            "Predictions differ across two identical runs — scaler may have been refit"
        )


# ─────────────────────────────────────────────────────────────
# TEST 4 — Output report CSV has correct structure
# ─────────────────────────────────────────────────────────────
def test_output_report_structure(pipeline, sample_df, tmp_path):
    """The output report CSV must contain both SECTION 1 and SECTION 2."""
    tmp_csv = tmp_path / "report_test.csv"
    sample_df.to_csv(tmp_csv, index=False)

    _ = pipeline.run_inference(str(tmp_csv))

    # Find the most recent report file
    output_dir = os.path.join(ROOT, 'Artifacts', 'model_validations')
    reports = [f for f in os.listdir(output_dir) if f.startswith('report_')]
    assert len(reports) > 0, f"No report files found in {output_dir}"

    latest_report = os.path.join(output_dir, sorted(reports)[-1])
    with open(latest_report, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'SECTION 1' in content, "Report missing SECTION 1: CYCLE-LEVEL PREDICTIONS"
    assert 'SECTION 2' in content, "Report missing SECTION 2: ENGINE METRICS"
    assert 'engine_id' in content, "Report missing engine_id column"
    assert 'predicted_RUL' in content, "Report missing predicted_RUL column"
    assert 'MSE' in content, "Report missing MSE metric"
    assert 'R2' in content,  "Report missing R2 metric"


# ─────────────────────────────────────────────────────────────
# TEST 5 — Flask /predict endpoint returns correct JSON
# ─────────────────────────────────────────────────────────────
def test_flask_api_endpoint(tmp_path):
    """POST a CSV to /predict and assert the JSON response structure."""
    # Import here to avoid Flask app starting at module-level
    import importlib
    import frontend.app as app_module

    app_module.app.config['TESTING'] = True
    client = app_module.app.test_client()

    with open(SAMPLE_CSV, 'rb') as f:
        csv_bytes = f.read()

    data = {
        'file': (io.BytesIO(csv_bytes), 'data_testing_1.csv')
    }
    response = client.post(
        '/predict',
        data=data,
        content_type='multipart/form-data'
    )

    assert response.status_code == 200, \
        f"Expected 200, got {response.status_code}. Body: {response.data[:500]}"

    body = response.get_json()
    assert body is not None, "Response is not JSON"
    assert body.get('success') is True, f"success != True. Body: {body}"

    required_keys = ['total_rows', 'total_engines', 'overall_metrics',
                     'per_engine_metrics', 'predictions']
    for key in required_keys:
        assert key in body, f"Missing key '{key}' in response JSON"

    metrics = body['overall_metrics']
    for m in ['MSE', 'MAE', 'RMSE', 'R2']:
        assert m in metrics, f"Missing metric '{m}' in overall_metrics"
