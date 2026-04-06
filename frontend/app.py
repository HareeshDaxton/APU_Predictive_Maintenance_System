"""
Flask backend for APU Predictive Maintenance Frontend.

LOCAL MODE  : loads model/scaler from Artifacts/ folder
GCP MODE    : downloads model/scaler from Google Cloud Storage at startup,
              saves every uploaded CSV to GCS, and logs predictions to Cloud SQL.

Environment variables (set in Cloud Run):
  GCS_MODEL_BUCKET   — GCS bucket containing model_LGBM.pkl, scaler.pkl, baseline_stats.json
  GCS_DATA_BUCKET    — GCS bucket where uploaded CSVs are saved
  GCP_PROJECT_ID     — GCP project ID (used for Secret Manager + Cloud Monitoring)
  DB_NAME            — Cloud SQL database name (default: apu_predictions)
  DB_USER            — Cloud SQL user (default: apu_user)
  PORT               — port override (default: 8080)
"""

import os
import sys
import json
import tempfile
import io
import logging

import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Force UTF-8 on Windows ────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── Project root on sys.path ──────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.Pipelines.inference_pipeline import InferencePipeline

app = Flask(__name__)
CORS(app)

# ── Environment detection ─────────────────────────────────────
GCP_MODE         = os.environ.get('GCP_MODE', 'False').lower() == 'true'
GCS_MODEL_BUCKET = os.environ.get('GCS_MODEL_BUCKET', 'apu-model-artifacts-apu-predictive-maintenance')
GCS_DATA_BUCKET  = os.environ.get('GCS_DATA_BUCKET', '')
GCP_PROJECT_ID   = os.environ.get('GCP_PROJECT_ID', '')
DB_NAME          = os.environ.get('DB_NAME', 'apu_predictions')
DB_USER          = os.environ.get('DB_USER', 'apu_user')
IS_GCP           = GCP_MODE or ('GCS_MODEL_BUCKET' in os.environ)

# ── Artifact paths ────────────────────────────────────────────
if IS_GCP:
    MODEL_PATH  = '/tmp/model_LGBM.pkl'
    SCALER_PATH = '/tmp/scaler.pkl'
    BASELINE_PATH = '/tmp/baseline_stats.json'
else:
    MODEL_PATH  = os.path.join(ROOT, 'Artifacts', 'Model', 'model_LGBM.pkl')
    SCALER_PATH = os.path.join(ROOT, 'Artifacts', 'Scaler', 'scaler.pkl')
    BASELINE_PATH = os.path.join(ROOT, 'Artifacts', 'baseline_stats.json')

# ── GCS helpers ───────────────────────────────────────────────
def _gcs_download(bucket_name: str, blob_name: str, local_path: str) -> None:
    """Download a file from GCS to a local path."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logging.info(f'Downloaded gs://{bucket_name}/{blob_name} -> {local_path}')


def _gcs_upload_bytes(bucket_name: str, blob_name: str, data: bytes) -> None:
    """Upload raw bytes to GCS."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data)
        logging.info(f'Uploaded to gs://{bucket_name}/{blob_name}')
    except Exception as e:
        logging.warning(f'GCS upload failed (non-fatal): {e}')


# ── At startup: download model files from GCS if running in GCP ──
if IS_GCP:
    logging.info('GCP mode detected — downloading model artifacts from GCS...')
    try:
        _gcs_download(GCS_MODEL_BUCKET, 'model_LGBM.pkl', MODEL_PATH)
        _gcs_download(GCS_MODEL_BUCKET, 'scaler.pkl', SCALER_PATH)
        _gcs_download(GCS_MODEL_BUCKET, 'baseline_stats.json', BASELINE_PATH)
        logging.info('Model artifacts downloaded successfully.')
    except Exception as e:
        logging.error(f'Failed to download model artifacts from GCS: {e}')
        # App will still start — inference will fail gracefully per-request


# ── Database helpers ──────────────────────────────────────────
def _get_db_password() -> str:
    """Fetch DB password from GCP Secret Manager."""
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f'projects/{GCP_PROJECT_ID}/secrets/db-password/versions/latest'
    response = client.access_secret_version(request={'name': name})
    return response.payload.data.decode('UTF-8')


def _get_db_connection():
    """Return a psycopg2 connection to Cloud SQL (GCP mode only)."""
    import psycopg2
    password = _get_db_password()
    db_host  = os.environ.get('DB_HOST', '127.0.0.1')
    return psycopg2.connect(
        host=db_host,
        database=DB_NAME,
        user=DB_USER,
        password=password
    )


def _save_predictions_to_db(predictions_df: pd.DataFrame,
                             model_version: str,
                             filename: str) -> None:
    """Write per-cycle predictions to Cloud SQL predictions table."""
    try:
        conn   = _get_db_connection()
        cursor = conn.cursor()
        for _, row in predictions_df.iterrows():
            cursor.execute(
                """
                INSERT INTO predictions
                  (engine_id, cycle, predicted_rul, true_rul,
                   timestamp, model_version, filename)
                VALUES (%s, %s, %s, %s, NOW(), %s, %s)
                """,
                (int(row['engine_id']), int(row['cycle']),
                 float(row['predicted_RUL']), float(row['true_RUL']),
                 model_version, filename)
            )
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f'Saved {len(predictions_df)} predictions to Cloud SQL.')
    except Exception as e:
        logging.warning(f'Cloud SQL write failed (non-fatal): {e}')


# ── Cloud Monitoring helpers ───────────────────────────────────
def _log_custom_metric(metric_name: str, value: float) -> None:
    """Log a custom metric to GCP Cloud Monitoring."""
    if not IS_GCP or not GCP_PROJECT_ID:
        return
    try:
        import time
        from google.cloud import monitoring_v3
        client = monitoring_v3.MetricServiceClient()
        project_name = f'projects/{GCP_PROJECT_ID}'
        series = monitoring_v3.TimeSeries()
        series.metric.type = f'custom.googleapis.com/apu/{metric_name}'
        series.resource.type = 'global'
        point = monitoring_v3.Point()
        point.value.double_value = value
        point.interval.end_time.seconds = int(time.time())
        series.points = [point]
        client.create_time_series(
            request={'name': project_name, 'time_series': [series]}
        )
    except Exception as e:
        logging.warning(f'Cloud Monitoring log failed (non-fatal): {e}')


# ── Lazy pipeline singleton ───────────────────────────────────
_pipeline = None

def _get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline(
            config_path=os.path.join(ROOT, 'config', 'config.yaml')
        )
        if IS_GCP:
            _pipeline.model_path = MODEL_PATH
            _pipeline.scaler_path = SCALER_PATH
            _pipeline.baseline_stats_path = BASELINE_PATH
    return _pipeline


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    try:
        # ── Read uploaded bytes ────────────────────────────────
        file_bytes = file.read()
        filename   = file.filename

        # ── Save uploaded CSV to GCS (if in GCP mode) ─────────
        if IS_GCP and GCS_DATA_BUCKET:
            from datetime import datetime
            ts = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
            _gcs_upload_bytes(
                GCS_DATA_BUCKET,
                f'uploads/{ts}_{filename}',
                file_bytes
            )

        # ── Write to temp file for inference pipeline ──────────
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False, mode='wb'
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # ── Run inference ──────────────────────────────────────
        pl = _get_pipeline()
        results_df = pl.run_inference(tmp_path)
        os.unlink(tmp_path)

        # ── Compute metrics ────────────────────────────────────
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        import numpy as np

        true_rul = results_df['true_RUL']
        pred_rul = results_df['predicted_RUL']

        mse  = round(float(mean_squared_error(true_rul, pred_rul)), 4)
        mae  = round(float(mean_absolute_error(true_rul, pred_rul)), 4)
        rmse = round(float(np.sqrt(mse)), 4)
        r2   = round(float(r2_score(true_rul, pred_rul)), 4)

        # ── Per-engine metrics ─────────────────────────────────
        per_engine = []
        for eid, grp in results_df.groupby('engine_id'):
            g_mse  = round(float(mean_squared_error(grp['true_RUL'], grp['predicted_RUL'])), 4)
            g_mae  = round(float(mean_absolute_error(grp['true_RUL'], grp['predicted_RUL'])), 4)
            g_rmse = round(float(np.sqrt(g_mse)), 4)
            g_r2   = round(float(r2_score(grp['true_RUL'], grp['predicted_RUL'])), 4)
            per_engine.append({
                'engine_id': int(eid),
                'num_cycles': len(grp),
                'MSE': g_mse, 'MAE': g_mae, 'RMSE': g_rmse, 'R2': g_r2
            })

        # ── Serialize predictions ──────────────────────────────
        predictions = results_df.to_dict(orient='records')
        for p in predictions:
            for k, v in p.items():
                if hasattr(v, 'item'):
                    p[k] = v.item()

        # ── Save to Cloud SQL (GCP mode) ───────────────────────
        if IS_GCP:
            _save_predictions_to_db(results_df, model_version='v1', filename=filename)

        # ── Log custom metrics to Cloud Monitoring ─────────────
        _log_custom_metric('r2_score', r2)
        _log_custom_metric('inference_row_count', float(len(results_df)))

        # ── Get drift report if available ─────────────────────
        drift_report = getattr(results_df, '_drift_report', None)

        return jsonify({
            'success':          True,
            'total_rows':       len(results_df),
            'total_engines':    results_df['engine_id'].nunique(),
            'overall_metrics':  {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2},
            'per_engine_metrics': per_engine,
            'predictions':      predictions,
        })

    except Exception as e:
        logging.exception('Prediction error')
        return jsonify({'error': str(e)}), 500


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print('\n' + '=' * 60)
    print('  APU Predictive Maintenance - Frontend Server')
    print(f'  http://0.0.0.0:{port}')
    print(f'  GCP Mode: {IS_GCP}')
    print('=' * 60 + '\n')
    app.run(host='0.0.0.0', port=port, debug=not IS_GCP)
