import os
from typing import Tuple
import joblib
import numpy as np
import sys
import yaml
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from Utils.Exception.exception import CustomException
from Utils.Logging.logger import logging


class FeatureEngineering:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.train_data_path = self.config['data']['train_data_path']
        self.test_data_path = self.config['data']['test_data_path']
        
        self.preprocessed_train_path = self.config['data']['train_preprocessed_path']
        self.preprocessed_test_path = self.config['data']['test_preprocessed_path']
        
        self.scaler_path = self.config['data']['scaler_path']
        
        self.sensor_columns = [
            'sensor_2','sensor_3','sensor_4','sensor_7','sensor_8',
            'sensor_9','sensor_11','sensor_12','sensor_13','sensor_14',
            'sensor_15','sensor_17','sensor_20','sensor_21'
        ]
        
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CustomException(f"Error loading config file: {e}")
        
    def _add_rul(self, df : pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Adding ReLU features...")
            
            max_cycle = df.groupby('engine_id')['cycle'].transform('max')
            df['RUL'] = max_cycle - df['cycle']
            
            logging.info("ReLU features added successfully.")
            
            return df
        
        except Exception as e:
            logging.error('The error occurred in <src -> Data_preprocessing -> Feature_engineering.py -> _add_rul()>' )
            raise CustomException(f"Error adding ReLU features: {e}")


# To make sure the model learns engine health, not differences caused by operating modes.
# We normalize operating conditions so the model focuses on degradation, not how the engine is being used.
    def _normalize_operating_conditions(self, df : pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Normalizing operating conditions...")
            
            op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
            df[self.sensor_columns] = df.groupby(op_cols)[self.sensor_columns]\
            .transform(lambda x: x - x.mean())

            logging.info("Operating conditions normalized successfully.")
            
            return df
        
        except Exception as e:
            logging.error('The error occurred in <src -> Data_preprocessing -> Feature_engineering.py -> _normalize_operating_conditions()>' )
            raise CustomException(f"Error normalizing operating conditions: {e}")
        
        
# It converts raw cycle number into a percentage of engine life used.
# Standardize lifecycle progression across engines so the model learns degradation stage instead of raw time.
    def _add_cycle_normalization(self, df : pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Adding cycle normalization...")
            
            max_cycle = df.groupby('engine_id')['cycle'].transform('max')
            df['cycle_normalized'] = df['cycle'] / max_cycle
            
            logging.info("Cycle normalization added successfully.")
            
            return df
        
        except Exception as e:
            logging.error('The error occurred in <src -> Data_preprocessing -> Feature_engineering.py -> _add_cycle_normalization()>' )
            raise CustomException(f"Error adding cycle normalization: {e}")
        
    
#Rolling standard deviation measures recent sensor instability, helping detect early signs of mechanical failure.
    def _add_rolling_features(self, df: pd.DataFrame, window=5) -> pd.DataFrame:
        try:
            logging.info("Adding rolling features...")
            
            for col in self.sensor_columns:
                
                df[f"{col}_rolling_mean"] = df.groupby('engine_id')[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
                
                df[f"{col}_rolling_std"] = df.groupby("engine_id")[col].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True)
                                 
                # trend feature (change direction) 
                df[f"{col}_trend"] = df.groupby('engine_id')[col].diff()
                                  
            df.fillna(0, inplace=True)  # handle NaN values from rolling calculations
            logging.info("Rolling features added successfully.")

            return df

        except Exception as e:
            logging.error('The error occurred in <src -> Data_preprocessing -> Feature_engineering.py -> _add_rolling_features()>' )
            raise CustomException(f"Error adding rolling features: {e}")
        
    


    def _scal_scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Scaling features")
            
            exclude_cols = [
            "engine_id",
            "RUL",
            "fault_label",
            "fault_type",
            "fault_target"
            ]

            feature_cols = [col for col in train_df.columns if col not in exclude_cols]
            
            scaler = StandardScaler()

            logging.info('Scaling train_df ')
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            
            logging.info('Scaling test_df')
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            
            return train_df, test_df
        
        
        except Exception as e:
            logging.error('The error occurred in <src -> Data_preprocessing -> Feature_engineering.py -> _scal_scale_features()>' )
            raise CustomException(f"Error scaling features: {e}")



    def run_feature_engineering(self):
        try:
            logging.info("Starting feature engineering process...")
            
            logging.info("Loading data...")
            
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)
            
            logging.info("Data loaded successfully.")
            
            logging.info("Adding RUL features...")
            
            train_df = self._add_rul(train_df)
            test_df = self._add_rul(test_df)
            
            logging.info('RUL features added successfully.')
            
            logging.info("Normalizing operating conditions...")
            
            train_df = self._normalize_operating_conditions(train_df)
            test_df = self._normalize_operating_conditions(test_df)
            
            logging.info("Normalizing operating conditions completed successfully.")
            
            
            logging.info("Adding cycle normalization...")
            
            train_df = self._add_cycle_normalization(train_df)
            test_df = self._add_cycle_normalization(test_df)
            
            logging.info("Cycle normalization added successfully.") 
            
            logging.info("Adding rolling features...")
            
            train_df = self._add_rolling_features(train_df)
            test_df = self._add_rolling_features(test_df)
            
            logging.info("Rolling features added successfully.")
            
            logging.info("Scaling features...")
            
            train_df, test_df = self._scal_scale_features(train_df, test_df)
            
            os.makedirs(os.path.dirname(self.preprocessed_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.preprocessed_test_path), exist_ok=True)
            
            train_df.to_csv(self.preprocessed_train_path, index=False)
            test_df.to_csv(self.preprocessed_test_path, index=False)

            logging.info("Feature engineering process completed successfully.")
            
        except Exception as e: 
            logging.error('The error occurred in <src -> Data_preprocessing -> Feature_engineering.py -> run_feature_engineering()>' )
            raise CustomException(f"Error in feature engineering process: {e}")