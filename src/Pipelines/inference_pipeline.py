import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Utils.Logging.logger import logging
from Utils.Exception.exception import CustomException


class InferencePipeline:
    """
    Inference Pipeline for APU Predictive Maintenance.

    Accepts a new unseen CSV file (any number of rows / engines),
    applies the same preprocessing used during training, runs the
    saved LightGBM model, and saves predictions + per-engine metrics
    to Artifacts/model_validations/.

    Input CSV required columns:
        engine_id, cycle, op_setting_1, op_setting_2, op_setting_3,
        sensor_1 ... sensor_21

    Optional columns (ignored if present):
        anomaly_score, fault_label, fault_type, fault_target
    """

    # ------------------------------------------------------------------ #
    #  Same 14 informative sensors used during training                   #
    # ------------------------------------------------------------------ #
    SENSOR_COLUMNS = [
        'sensor_2',  'sensor_3',  'sensor_4',  'sensor_7',
        'sensor_8',  'sensor_9',  'sensor_11', 'sensor_12',
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
        'sensor_20', 'sensor_21'
    ]

    # Columns that are NOT features (excluded from scaling)
    EXCLUDE_FROM_SCALING = [
        'engine_id', 'RUL', 'fault_label', 'fault_type', 'fault_target'
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config      = self._load_config()

        # Paths from config
        self.model_path  = "Artifacts/Model/model_LGBM.pkl"
        self.scaler_path = self.config['data']['scaler_path']

        # Output directory for predictions & metrics
        self.output_dir  = "Artifacts/model_validations"

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CustomException(f"Failed to load config: {e}", sys)

    def _load_artifacts(self):
        """Load the saved scaler and trained model from disk."""
        try:
            logging.info("Loading saved scaler and model...")
            scaler = joblib.load(self.scaler_path)
            model  = joblib.load(self.model_path)
            logging.info("Scaler and model loaded successfully.")
            return scaler, model
        except Exception as e:
            logging.error("Failed to load model/scaler artifacts.")
            raise CustomException(f"Failed to load artifacts: {e}", sys)

    def _load_input_csv(self, input_path: str) -> pd.DataFrame:
        """Load and validate the input CSV file."""
        try:
            logging.info(f"Loading input CSV: {input_path}")
            df = pd.read_csv(input_path)
            logging.info(f"Input shape: {df.shape}  |  Engines: {df['engine_id'].nunique()}")

            # Validate required columns
            required = (
                ['engine_id', 'cycle',
                 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
                [f'sensor_{i}' for i in range(1, 22)]
            )
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Input CSV is missing required columns: {missing}"
                )

            # Sort by engine then cycle (critical for rolling features)
            df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
            return df

        except Exception as e:
            logging.error("Error loading input CSV.")
            raise CustomException(f"Error loading input CSV: {e}", sys)

    # ------------------------------------------------------------------ #
    #  Preprocessing steps — MUST mirror feature_engineering_pipeline.py  #
    # ------------------------------------------------------------------ #

    def _compute_proxy_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RUL using the maximum observed cycle per engine as proxy.

        For fully observed engines (training-style data), this is exact.
        For partial histories, it gives the relative remaining cycles
        within the provided window, which is still a valid relative
        degradation signal for the model.
        """
        try:
            logging.info("Computing proxy RUL (max_cycle - current_cycle)...")
            max_cycle = df.groupby('engine_id')['cycle'].transform('max')
            df['RUL'] = max_cycle - df['cycle']
            logging.info("Proxy RUL computed.")
            return df
        except Exception as e:
            raise CustomException(f"Error computing RUL: {e}", sys)

    def _normalize_operating_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean-center sensors within each operating condition group."""
        try:
            logging.info("Normalizing operating conditions...")
            op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
            df[self.SENSOR_COLUMNS] = (
                df.groupby(op_cols)[self.SENSOR_COLUMNS]
                  .transform(lambda x: x - x.mean())
            )
            logging.info("Operating conditions normalized.")
            return df
        except Exception as e:
            raise CustomException(
                f"Error normalizing operating conditions: {e}", sys
            )

    def _add_cycle_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """cycle_normalized = cycle / max_cycle_per_engine (0–1 scale)."""
        try:
            logging.info("Adding cycle normalization...")
            max_cycle = df.groupby('engine_id')['cycle'].transform('max')
            df['cycle_normalized'] = df['cycle'] / max_cycle
            logging.info("Cycle normalization added.")
            return df
        except Exception as e:
            raise CustomException(
                f"Error adding cycle normalization: {e}", sys
            )

    def _add_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add rolling mean, rolling std, and trend (diff) per sensor per engine."""
        try:
            logging.info(f"Adding rolling features (window={window})...")
            for col in self.SENSOR_COLUMNS:
                df[f"{col}_rolling_mean"] = (
                    df.groupby('engine_id')[col]
                      .rolling(window, min_periods=1)
                      .mean()
                      .reset_index(level=0, drop=True)
                )
                df[f"{col}_rolling_std"] = (
                    df.groupby('engine_id')[col]
                      .rolling(window, min_periods=1)
                      .std()
                      .reset_index(level=0, drop=True)
                )
                df[f"{col}_trend"] = df.groupby('engine_id')[col].diff()

            df.fillna(0, inplace=True)
            logging.info("Rolling features added.")
            return df
        except Exception as e:
            raise CustomException(f"Error adding rolling features: {e}", sys)

    def _apply_scaler(self, df: pd.DataFrame, scaler) -> pd.DataFrame:
        """Apply the saved StandardScaler (transform only — never refit)."""
        try:
            logging.info("Applying saved StandardScaler...")

            # If anomaly_score is missing (input has no fault injection),
            # add it as 0 — neutral value meaning "no anomaly detected".
            # The scaler was fitted on training data that included this column.
            if 'anomaly_score' not in df.columns:
                logging.info("anomaly_score not found in input — defaulting to 0 (no anomaly).")
                df['anomaly_score'] = 0.0

            # Dynamically identify optional fault columns to exclude
            extra_exclude = [
                c for c in ['fault_label', 'fault_type', 'fault_target']
                if c in df.columns
            ]
            exclude = self.EXCLUDE_FROM_SCALING + extra_exclude

            feature_cols = [c for c in df.columns if c not in exclude]

            df[feature_cols] = scaler.transform(df[feature_cols])
            logging.info("Scaler applied.")
            return df, feature_cols
        except Exception as e:
            raise CustomException(f"Error applying scaler: {e}", sys)

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that were not used as model input features.

        NOTE: anomaly_score is intentionally KEPT here — the model was
        trained with it as an input feature. For clean input data without
        fault injection, it defaults to 0.0 (set in _apply_scaler).
        """
        drop_cols = [
            'engine_id', 'RUL',
            'fault_label', 'fault_type', 'fault_target'
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]
        return df.drop(columns=drop_cols)

    # ------------------------------------------------------------------ #
    #  Metrics helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_metrics(self, true_rul: pd.Series,
                          pred_rul: pd.Series) -> dict:
        """Return MSE, MAE, RMSE, R2 for a pair of series."""
        mse  = mean_squared_error(true_rul, pred_rul)
        mae  = mean_absolute_error(true_rul, pred_rul)
        rmse = np.sqrt(mse)
        r2   = r2_score(true_rul, pred_rul)
        return {'MSE': round(mse, 4), 'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

    # ------------------------------------------------------------------ #
    #  Save outputs                                                        #
    # ------------------------------------------------------------------ #

    def _save_outputs(self, predictions_df: pd.DataFrame,
                      metrics_df: pd.DataFrame,
                      overall_metrics: dict) -> str:
        """
        Save a SINGLE combined report CSV to Artifacts/model_validations/:
          report_TIMESTAMP.csv
          ├── Section 1: Per-cycle predictions (engine_id, cycle, true_RUL, predicted_RUL)
          ├── [blank separator row]
          └── Section 2: Per-engine + OVERALL metrics
        Returns the saved report path.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            report_path = os.path.join(
                self.output_dir, f"report_{timestamp}.csv"
            )

            # Build overall metrics row
            overall_row = pd.DataFrame([{
                'engine_id': 'OVERALL',
                'num_cycles': len(predictions_df),
                **overall_metrics
            }])
            full_metrics_df = pd.concat(
                [metrics_df, overall_row], ignore_index=True
            )

            # Write: predictions → blank line → metrics, all in one file
            with open(report_path, 'w', newline='', encoding='utf-8') as f:
                f.write("# ── SECTION 1: CYCLE-LEVEL PREDICTIONS ──\n")
                predictions_df.to_csv(f, index=False)
                f.write("\n# ── SECTION 2: ENGINE METRICS ──\n")
                full_metrics_df.to_csv(f, index=False)

            logging.info(f"Combined report saved -> {report_path}")

            print(f"\n{'='*60}")
            print(" INFERENCE COMPLETE")
            print(f"{'='*60}")
            print(f"  Report       -> {report_path}")
            print(f"\n  OVERALL METRICS:")
            print(f"    MSE  : {overall_metrics['MSE']}")
            print(f"    MAE  : {overall_metrics['MAE']}")
            print(f"    RMSE : {overall_metrics['RMSE']}")
            print(f"    R2   : {overall_metrics['R2']}")
            print(f"{'='*60}\n")

            return report_path

        except Exception as e:
            raise CustomException(f"Error saving outputs: {e}", sys)

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run_inference(self, input_csv_path: str) -> pd.DataFrame:
        """
        Full inference pipeline.

        Args:
            input_csv_path: Absolute or relative path to the input CSV file.

        Returns:
            DataFrame with columns: engine_id, cycle, true_RUL, predicted_RUL
        """
        try:
            logging.info("="*60)
            logging.info("INFERENCE PIPELINE STARTED")
            logging.info(f"Input file: {input_csv_path}")
            logging.info("="*60)

            # 1. Load artifacts
            scaler, model = self._load_artifacts()

            # 2. Load & validate input CSV
            df = self._load_input_csv(input_csv_path)

            # Keep a clean copy of identifiers BEFORE any transformation
            id_df = df[['engine_id', 'cycle']].copy()

            # 3. Compute proxy RUL (used for metrics; also added as a column
            #    because the scaler expects it — it is excluded from features)
            df = self._compute_proxy_rul(df)
            true_rul = df['RUL'].copy()

            # 4. Normalize operating conditions
            df = self._normalize_operating_conditions(df)

            # 5. Cycle normalization
            df = self._add_cycle_normalization(df)

            # 6. Rolling features
            df = self._add_rolling_features(df)

            # 7. Apply saved scaler
            df, _ = self._apply_scaler(df, scaler)

            # 8. Select model input features (drop id + target cols)
            X = self._select_features(df)

            # 9. Predict
            logging.info(f"Running model inference on {len(X)} rows...")
            predicted_rul = model.predict(X)
            logging.info("Inference complete.")

            # 10. Build predictions dataframe
            predictions_df = id_df.copy()
            predictions_df['true_RUL']      = true_rul.values
            predictions_df['predicted_RUL'] = np.round(predicted_rul, 2)

            # 11. Compute per-engine metrics
            per_engine_metrics = []
            for engine_id, grp in predictions_df.groupby('engine_id'):
                m = self._compute_metrics(grp['true_RUL'], grp['predicted_RUL'])
                per_engine_metrics.append({
                    'engine_id' : engine_id,
                    'num_cycles': len(grp),
                    **m
                })
            metrics_df = pd.DataFrame(per_engine_metrics)

            # 12. Overall metrics
            overall_metrics = self._compute_metrics(
                predictions_df['true_RUL'],
                predictions_df['predicted_RUL']
            )

            # 13. Save everything
            self._save_outputs(predictions_df, metrics_df, overall_metrics)

            logging.info("Inference pipeline completed successfully.")
            return predictions_df

        except Exception as e:
            logging.error("Error in inference pipeline.")
            raise CustomException(f"Inference pipeline failed: {e}", sys)
