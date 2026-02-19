import os
import numpy as np
import sys
import pandas as pd
import yaml
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
        
    

























# _normalize_operating_conditions() 
# To make sure the model learns engine health, not differences caused by operating modes.
# We normalize operating conditions so the model focuses on degradation, not how the engine is being used.

# _add_cycle_normalization
# It converts raw cycle number into a percentage of engine life used.
# Standardize lifecycle progression across engines so the model learns degradation stage instead of raw time.

#Rolling standard deviation measures recent sensor instability, helping detect early signs of mechanical failure.