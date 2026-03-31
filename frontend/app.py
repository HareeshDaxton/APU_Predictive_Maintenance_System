"""
Flask backend for APU Predictive Maintenance Frontend.
Exposes a /predict endpoint that accepts a CSV upload,
runs the InferencePipeline, and returns JSON results.
"""

import os
import sys
import json
import tempfile
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Force UTF-8 output on Windows to prevent charmap encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path so we can import our pipeline
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.Pipelines.inference_pipeline import InferencePipeline

app = Flask(__name__)
CORS(app)

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline(config_path=os.path.join(ROOT, 'config', 'config.yaml'))
    return pipeline


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

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='wb') as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        # Run inference
        pl = get_pipeline()
        results_df = pl.run_inference(tmp_path)

        # Compute metrics fresh from results
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        true_rul = results_df['true_RUL']
        pred_rul = results_df['predicted_RUL']

        mse  = round(float(mean_squared_error(true_rul, pred_rul)), 4)
        mae  = round(float(mean_absolute_error(true_rul, pred_rul)), 4)
        rmse = round(float(np.sqrt(mse)), 4)
        r2   = round(float(r2_score(true_rul, pred_rul)), 4)

        # Per-engine metrics
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

        # Build predictions rows
        predictions = results_df.to_dict(orient='records')
        for p in predictions:
            for k, v in p.items():
                if hasattr(v, 'item'):
                    p[k] = v.item()

        os.unlink(tmp_path)

        return jsonify({
            'success': True,
            'total_rows': len(results_df),
            'total_engines': results_df['engine_id'].nunique(),
            'overall_metrics': {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2},
            'per_engine_metrics': per_engine,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  APU Predictive Maintenance — Frontend Server")
    print("  http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
