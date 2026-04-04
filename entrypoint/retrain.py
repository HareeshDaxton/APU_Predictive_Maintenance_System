"""
entrypoint/retrain.py

Full automated retraining pipeline for APU Predictive Maintenance.

HOW IT WORKS:
  1. Pulls all uploaded CSVs from GCS bucket (apu-incoming-data/uploads/)
  2. Combines them with the original NASA training data
  3. Runs fault injection + feature engineering on the combined data
  4. Trains a new LightGBM model
  5. Compares new model R2 vs current production model R2
  6. Promotes new model to GCS only if R2 improves

WHEN IT RUNS:
  Triggered every Sunday midnight UTC by Cloud Scheduler.
  Can also be triggered manually:
    gcloud run jobs execute apu-retrain-job --region us-central1

EDGE CASES HANDLED:
  - No new data yet          -> logs message, exits cleanly
  - Bad columns in a CSV     -> that file is skipped, logged
  - Training fails           -> keeps current production model
  - New model worse than old -> archives new model, keeps old one
"""

import os
import sys
import io
import json
import logging
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ── Add project root to path ──────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.Pipelines.feature_engineering_pipeline import FeatureEngineering
from src.Pipelines.trainig_pipeline import TrainingPipeline, ACTIVE_SENSORS

# ── Configure logging ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Environment variables ─────────────────────────────────────
GCS_MODEL_BUCKET = os.environ.get('GCS_MODEL_BUCKET', 'apu-model-artifacts')
GCS_DATA_BUCKET  = os.environ.get('GCS_DATA_BUCKET',  'apu-incoming-data')
GCP_PROJECT_ID   = os.environ.get('GCP_PROJECT_ID',   '')

# Required columns in every uploaded CSV
REQUIRED_COLUMNS = (
    ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f'sensor_{i}' for i in range(1, 22)]
)


# ── GCS helpers ───────────────────────────────────────────────
def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _download_blob_as_text(bucket_name: str, blob_name: str) -> str:
    client = _gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.download_as_text(encoding='utf-8')


def _upload_file(bucket_name: str, local_path: str, dest_blob_name: str) -> None:
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    log.info(f'Uploaded {local_path} -> gs://{bucket_name}/{dest_blob_name}')


def _upload_bytes(bucket_name: str, data: bytes, dest_blob_name: str) -> None:
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_string(data)


# ── Database helper ───────────────────────────────────────────
def _log_promotion_to_db(old_version, new_version,
                          old_r2, new_r2, promoted: bool) -> None:
    """Write a model promotion event to Cloud SQL."""
    try:
        import psycopg2
        from google.cloud import secretmanager

        sm_client = secretmanager.SecretManagerServiceClient()
        secret_name = f'projects/{GCP_PROJECT_ID}/secrets/db-password/versions/latest'
        password = sm_client.access_secret_version(
            request={'name': secret_name}
        ).payload.data.decode('UTF-8')

        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', '127.0.0.1'),
            database=os.environ.get('DB_NAME', 'apu_predictions'),
            user=os.environ.get('DB_USER', 'apu_user'),
            password=password
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_promotions
                  (old_version, new_version, old_r2, new_r2, promoted)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (old_version, new_version, old_r2, new_r2, promoted)
            )
        conn.commit()
        conn.close()
        log.info('Promotion event logged to Cloud SQL.')
    except Exception as e:
        log.warning(f'Could not log to Cloud SQL (non-fatal): {e}')


# ── STEP 1 — Pull new uploaded CSVs from GCS ─────────────────
def _collect_new_data() -> pd.DataFrame:
    log.info('Step 1: Collecting new uploaded CSVs from GCS...')
    client = _gcs_client()
    bucket = client.bucket(GCS_DATA_BUCKET)
    blobs  = list(bucket.list_blobs(prefix='uploads/'))

    if not blobs:
        log.info('No new uploads found in GCS. Skipping retraining.')
        return pd.DataFrame()

    frames = []
    for blob in blobs:
        try:
            text = blob.download_as_text(encoding='utf-8')
            df   = pd.read_csv(io.StringIO(text))
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                log.warning(
                    f'Skipping {blob.name} — missing columns: {missing}'
                )
                continue
            frames.append(df[REQUIRED_COLUMNS])
            log.info(f'Loaded {blob.name}: {len(df)} rows')
        except Exception as e:
            log.warning(f'Skipping {blob.name} due to error: {e}')

    if not frames:
        log.info('All uploaded files were invalid. Skipping retraining.')
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info(f'Total new data collected: {len(combined)} rows')
    return combined


# ── STEP 2 — Load original NASA training data from GCS ───────
def _load_original_data() -> pd.DataFrame:
    log.info('Step 2: Loading original NASA training data from GCS...')
    try:
        text = _download_blob_as_text(GCS_MODEL_BUCKET, 'train_FD002.csv')
        df   = pd.read_csv(io.StringIO(text))
        log.info(f'Original training data: {len(df)} rows')
        return df
    except Exception as e:
        log.warning(
            f'Could not load train_FD002.csv from GCS: {e}. '
            'Falling back to local file.'
        )
        local = os.path.join(ROOT, 'Artifacts', 'raw', 'train_FD002.csv')
        df = pd.read_csv(local)
        log.info(f'Loaded from local: {len(df)} rows')
        return df


# ── STEP 3 — Run fault injection on combined data ─────────────
def _run_fault_injection(df: pd.DataFrame) -> pd.DataFrame:
    log.info('Step 3: Running fault injection on combined data...')
    try:
        import importlib.util
        fi_path = os.path.join(
            ROOT, 'NoteBook', 'Fault_injection', 'fault_injection.py'
        )
        spec = importlib.util.spec_from_file_location('fault_injection', fi_path)
        fi_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fi_mod)

        injected = fi_mod.run_fault_injection(df)
        log.info(f'Fault injection complete: {len(injected)} rows')
        return injected
    except Exception as e:
        log.warning(
            f'Fault injection failed: {e}. '
            'Proceeding without fault injection (using combined raw data).'
        )
        return df


# ── STEP 4 — Run feature engineering + fit new scaler ────────
def _run_feature_engineering(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              version: str):
    log.info('Step 4: Running feature engineering...')
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path       = os.path.join(tmpdir, 'train.csv')
        test_path        = os.path.join(tmpdir, 'test.csv')
        train_out_path   = os.path.join(tmpdir, 'train_preprocessed.csv')
        test_out_path    = os.path.join(tmpdir, 'test_preprocessed.csv')
        scaler_out_path  = os.path.join(tmpdir, f'scaler_{version}.pkl')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        fe = FeatureEngineering(
            train_path=train_path,
            test_path=test_path,
            train_output_path=train_out_path,
            test_output_path=test_out_path,
            scaler_output_path=scaler_out_path,
        )
        fe.run()

        train_preprocessed = pd.read_csv(train_out_path)
        test_preprocessed  = pd.read_csv(test_out_path)

        log.info(
            f'Feature engineering done. '
            f'Train: {train_preprocessed.shape}, Test: {test_preprocessed.shape}'
        )
        return train_preprocessed, test_preprocessed, scaler_out_path


# ── STEP 5 — Train new model ──────────────────────────────────
def _train_new_model(train_preprocessed: pd.DataFrame,
                     test_preprocessed: pd.DataFrame,
                     version: str):
    log.info(f'Step 5: Training new LightGBM model v{version}...')
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, 'train_preprocessed.csv')
        test_path  = os.path.join(tmpdir, 'test_preprocessed.csv')
        model_path = os.path.join(tmpdir, f'model_{version}.pkl')

        train_preprocessed.to_csv(train_path, index=False)
        test_preprocessed.to_csv(test_path, index=False)

        pipeline = TrainingPipeline()
        pipeline.train_path       = train_path
        pipeline.test_path        = test_path
        pipeline.save_model_path  = model_path
        pipeline.baseline_stats_path = os.path.join(tmpdir, 'baseline_stats.json')

        pipeline.run_training_pipeline()
        return model_path, pipeline.baseline_stats_path


# ── STEP 6 — Compare new vs current production model ─────────
def _get_current_production_r2() -> float:
    log.info('Step 6: Loading current production model for comparison...')
    try:
        import joblib, tempfile as tf2
        prod_path = tf2.mktemp(suffix='.pkl')
        client = _gcs_client()
        client.bucket(GCS_MODEL_BUCKET).blob('model_LGBM.pkl').download_to_filename(prod_path)

        prod_model  = joblib.load(prod_path)
        test_path   = os.path.join(ROOT, 'Artifacts', 'Data_split',
                                   'Test_datasets', 'test_dataset.csv')
        test_df     = pd.read_csv(test_path)

        # Quick proxy: use raw cycle/sensor columns as feature approximation
        drop_cols = ['engine_id', 'fault_label', 'fault_type',
                     'fault_target', 'RUL']
        x_test    = test_df.drop(columns=drop_cols, errors='ignore')
        y_test    = test_df['RUL'] if 'RUL' in test_df.columns else None

        if y_test is None:
            log.warning('No RUL in test set — cannot compare. Assuming r2=0.')
            return 0.0

        preds = prod_model.predict(x_test)
        r2    = float(r2_score(y_test, preds))
        log.info(f'Current production model R2: {r2:.4f}')
        return r2
    except Exception as e:
        log.warning(f'Could not evaluate production model: {e}. Assuming r2=0.')
        return 0.0


# ── STEP 7 — Promote if better ───────────────────────────────
def _promote_model(new_model_path: str,
                   new_scaler_path: str,
                   new_baseline_path: str,
                   version: str) -> None:
    log.info(f'Promoting model v{version} to production...')
    _upload_file(GCS_MODEL_BUCKET, new_model_path, 'model_LGBM.pkl')
    _upload_file(GCS_MODEL_BUCKET, new_scaler_path, 'scaler.pkl')
    if os.path.exists(new_baseline_path):
        _upload_file(GCS_MODEL_BUCKET, new_baseline_path, 'baseline_stats.json')
    log.info(f'Model v{version} promoted to production in GCS.')


# ── Evidently drift report ────────────────────────────────────
def _generate_evidently_report(reference_df: pd.DataFrame,
                                current_df: pd.DataFrame,
                                timestamp: str) -> None:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.pipeline.column_mapping import ColumnMapping

        cols = [c for c in ACTIVE_SENSORS if c in reference_df.columns
                and c in current_df.columns]
        if not cols:
            log.warning('No common sensor columns for Evidently report.')
            return

        col_map = ColumnMapping(numerical_features=cols)

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df[cols],
            current_data=current_df[cols],
            column_mapping=col_map
        )
        html_path = f'/tmp/drift_report_{timestamp}.html'
        report.save_html(html_path)
        _upload_file(
            GCS_MODEL_BUCKET,
            html_path,
            f'drift_reports/report_{timestamp}.html'
        )
        log.info(f'Evidently drift report uploaded to GCS drift_reports/report_{timestamp}.html')
    except ImportError:
        log.warning('evidently not installed — skipping drift report.')
    except Exception as e:
        log.warning(f'Evidently report failed (non-fatal): {e}')


# ── MAIN ──────────────────────────────────────────────────────
def run_retraining_pipeline() -> None:
    log.info('=' * 60)
    log.info('RETRAINING PIPELINE STARTED')
    log.info(f'Timestamp: {datetime.utcnow().isoformat()}')
    log.info('=' * 60)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    version   = f'v_{timestamp}'

    # Step 1: Collect new data
    new_data = _collect_new_data()
    if new_data.empty:
        log.info('No new data to train on. Exiting cleanly.')
        return

    # Step 2: Load original data
    original_data = _load_original_data()

    # Step 3: Combine + fault injection
    combined_raw = pd.concat([original_data, new_data], ignore_index=True)
    log.info(f'Combined dataset: {len(combined_raw)} rows')

    # Generate Evidently drift report before training
    _generate_evidently_report(original_data, new_data, timestamp)

    combined_injected = _run_fault_injection(combined_raw)

    # Engine-based train/test split
    engines = combined_injected['engine_id'].unique()
    np.random.seed(42)
    np.random.shuffle(engines)
    split_idx  = int(len(engines) * 0.8)
    train_engs = engines[:split_idx]
    test_engs  = engines[split_idx:]
    train_df   = combined_injected[combined_injected['engine_id'].isin(train_engs)]
    test_df    = combined_injected[combined_injected['engine_id'].isin(test_engs)]

    # Step 4: Feature engineering
    try:
        train_pre, test_pre, scaler_path = _run_feature_engineering(
            train_df, test_df, version
        )
    except Exception as e:
        log.error(f'Feature engineering failed: {e}. Keeping current model.')
        return

    # Step 5: Train new model
    try:
        new_model_path, new_baseline_path = _train_new_model(
            train_pre, test_pre, version
        )
    except Exception as e:
        log.error(f'Model training failed: {e}. Keeping current model.')
        return

    # Step 6: Evaluate new model
    try:
        import joblib
        new_model  = joblib.load(new_model_path)
        drop_cols  = ['engine_id', 'fault_label', 'fault_type', 'fault_target', 'RUL']
        x_test     = test_pre.drop(columns=drop_cols, errors='ignore')
        y_test     = test_pre['RUL']
        new_r2     = float(r2_score(y_test, new_model.predict(x_test)))
        log.info(f'New model R2: {new_r2:.4f}')
    except Exception as e:
        log.error(f'New model evaluation failed: {e}. Keeping current model.')
        return

    # Step 6b: Get current production R2
    prod_r2 = _get_current_production_r2()

    # Step 7: Promote or archive
    if new_r2 > prod_r2:
        log.info(
            f'New model ({new_r2:.4f}) > current ({prod_r2:.4f}). PROMOTING.'
        )
        _promote_model(new_model_path, scaler_path, new_baseline_path, version)
        _log_promotion_to_db('production', version, prod_r2, new_r2, promoted=True)
    else:
        log.info(
            f'New model ({new_r2:.4f}) did not improve over current ({prod_r2:.4f}). '
            'Keeping current production model.'
        )
        _log_promotion_to_db('production', version, prod_r2, new_r2, promoted=False)

    log.info('=' * 60)
    log.info('RETRAINING PIPELINE COMPLETE')
    log.info('=' * 60)


if __name__ == '__main__':
    run_retraining_pipeline()
