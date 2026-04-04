import os
import sys
import json
import yaml
import joblib
import subprocess
import pandas as pd
import numpy as np
from typing import Tuple
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.lightgbm

from Utils.Logging.logger import logging
from Utils.Exception.exception import CustomException

# Active sensor columns (must match inference pipeline exactly)
ACTIVE_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]


def _get_git_sha() -> str:
    """Return the current Git commit SHA (short). Returns 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha
    except Exception:
        return 'unknown'


class TrainingPipeline:
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()

        self.train_path = self.config['data']['train_preprocessed_path']
        self.test_path  = self.config['data']['test_preprocessed_path']
        self.model_params = self.config['model']['params']
        self.save_model_path = 'Artifacts/Model/model_LGBM.pkl'
        self.baseline_stats_path = 'Artifacts/baseline_stats.json'

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                logging.info('Loading configuration files...')
                return yaml.safe_load(f)
        except Exception as e:
            logging.error('Failed to load config files.')
            raise CustomException(f'Failed to load configuration files: {e}', sys)

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info('Loading train and test data...')
            train_df = pd.read_csv(self.train_path)
            test_df  = pd.read_csv(self.test_path)
            logging.info(f'Train shape: {train_df.shape} | Test shape: {test_df.shape}')
            return train_df, test_df
        except Exception as e:
            logging.error('Error loading data.')
            raise CustomException(f'Error loading data: {e}', sys)

    def _select_features(self, df: pd.DataFrame):
        try:
            logging.info('Selecting features for training...')
            drop_columns = ['engine_id', 'fault_label', 'fault_type', 'fault_target', 'RUL']
            x = df.drop(columns=drop_columns, errors='ignore')
            y = df['RUL']
            logging.info(f'Feature count: {x.shape[1]}')
            return x, y
        except Exception as e:
            logging.error('Error selecting features.')
            raise CustomException(f'Error selecting features: {e}', sys)

    def _train_model(self, x_train, y_train):
        try:
            logging.info('Training LightGBM model...')
            model = LGBMRegressor(**self.model_params)
            model.fit(x_train, y_train)
            logging.info('Model training completed.')
            return model
        except Exception as e:
            logging.error('Error during model training.')
            raise CustomException(f'Error during model training: {e}', sys)

    def _evaluate_model(self, model, x_test, y_test):
        try:
            logging.info('Evaluating model performance...')
            preds = model.predict(x_test)
            mse  = float(mean_squared_error(y_test, preds))
            mae  = float(mean_absolute_error(y_test, preds))
            r2   = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mse))
            logging.info(f'MSE={mse:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}')
            return mse, mae, r2, rmse
        except Exception as e:
            logging.error('Error evaluating model.')
            raise CustomException(f'Error evaluating model: {e}', sys)

    def _save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
            joblib.dump(model, self.save_model_path)
            logging.info(f'Model saved -> {self.save_model_path}')
        except Exception as e:
            logging.error('Error saving model.')
            raise CustomException(f'Error saving model: {e}', sys)

    def _save_baseline_stats(self, train_df: pd.DataFrame) -> None:
        """
        Compute mean and std for each of the 14 active sensors on the TRAINING
        data and save to Artifacts/baseline_stats.json.
        This file is used by the drift detection module at inference time.
        """
        try:
            logging.info('Computing baseline statistics for drift detection...')
            stats = {}
            for sensor in ACTIVE_SENSORS:
                if sensor in train_df.columns:
                    stats[sensor] = {
                        'mean': float(train_df[sensor].mean()),
                        'std':  float(train_df[sensor].std())
                    }
            os.makedirs(os.path.dirname(self.baseline_stats_path), exist_ok=True)
            with open(self.baseline_stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            logging.info(f'Baseline stats saved -> {self.baseline_stats_path}')
        except Exception as e:
            logging.error('Error saving baseline stats.')
            raise CustomException(f'Error saving baseline stats: {e}', sys)

    def run_training_pipeline(self):
        try:
            logging.info('=' * 60)
            logging.info('TRAINING PIPELINE STARTED')
            logging.info('=' * 60)

            train_df, test_df = self._load_data()

            x_train, y_train = self._select_features(train_df)
            x_test,  y_test  = self._select_features(test_df)

            logging.info(f'Training shape: {x_train.shape}')
            logging.info(f'Testing shape : {x_test.shape}')

            git_sha = _get_git_sha()

            # ── MLflow Experiment Tracking ──────────────────────────
            mlflow.set_experiment('APU_Predictive_Maintenance')

            with mlflow.start_run(run_name=f'APU_LightGBM_{git_sha}'):

                # Log hyperparameters
                mlflow.log_params(self.model_params)

                # Log dataset info
                mlflow.log_param('train_rows',    len(train_df))
                mlflow.log_param('test_rows',     len(test_df))
                mlflow.log_param('feature_count', x_train.shape[1])
                mlflow.log_param('train_path',    self.train_path)
                mlflow.log_param('git_sha',       git_sha)

                # Train
                model = self._train_model(x_train, y_train)

                # Evaluate
                mse, mae, r2, rmse = self._evaluate_model(model, x_test, y_test)

                # Log metrics
                mlflow.log_metrics({
                    'MSE':  round(mse,  4),
                    'MAE':  round(mae,  4),
                    'RMSE': round(rmse, 4),
                    'R2':   round(r2,   4),
                })

                # Log model artifact
                mlflow.lightgbm.log_model(model, artifact_path='model')
                mlflow.set_tag('model_stage', 'Staging')
                mlflow.set_tag('model_type',  'LGBMRegressor')

                logging.info(f'MLflow run logged. Git SHA: {git_sha}')

            # ── Save model and baseline stats ───────────────────────
            self._save_model(model)
            self._save_baseline_stats(train_df)

            logging.info('=' * 60)
            logging.info('TRAINING PIPELINE COMPLETE')
            logging.info(f'  MSE={mse:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}')
            logging.info('=' * 60)

        except Exception as e:
            logging.error('Error in training pipeline.')
            raise CustomException(e, sys)