import numpy as np
from Utils.Exception.exception import CustomException
from Utils.Logging.logger import logger
import yaml
import pandas as pd
import os
import sys

class DataSplit:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config  = self._load_config()

        self.data_path = self.config['data']['original_data_path']
        self.test_size = self.config['data']['test_size']   
        
        self.train_output_path = self.config['data']['train_data_path']
        self.test_output_path = self.config['data']['test_data_path']
        
    def _load_config(self):
            try:
                logger.info('Loading configuration')
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f'Error loading configuration =>src/Data_preprocessing/Data_split/Data_split.py => _load_config() => {e}')
                raise CustomException('Error loading configuration', sys)
            
            
# This splitting is done before feature engineering to prevent data leakage.
    def run_data_split(self):
            try:
                logger.info('Splitting data initiated')
                df = pd.read_csv(self.data_path)
                df = df.sort_values(by=['engine_id', 'cycle'])
                
                # Split by engine_id to prevent data leakage.
                # Ensures entire engine histories are kept either in train or test.
                # This simulates real-world scenario where model predicts on unseen engines.
                
                # Get unique engines
                unique_engines = df["engine_id"].unique()
                
                
                # Shuffle engines (industry standard)
                np.random.seed(42)                                              
                np.random.shuffle(unique_engines)       # NOTE only randomizing which engines go to train and test.    
                logger.info('"engine_id" are shuffeled')
                
                split_index = int(len(unique_engines) * (1 - self.test_size))
                
                logger.info("Splitting data into train and test sets based on engine_id")
                train_engines = unique_engines[:split_index]              
                test_engines = unique_engines[split_index:]
                logger.info("Splitting data is over")
                
                #Check id 'engine_id' is in train or test set and split accordingly
                train_df = df[df['engine_id'].isin(train_engines)]           
                test_df = df[df['engine_id'].isin(test_engines)]             
                                
                logger.info(f"Train engines: {len(train_engines)}")
                logger.info(f"Test engines: {len(test_engines)}")

                logger.info(f"Train rows: {train_df.shape[0]}")
                logger.info(f"Test rows: {test_df.shape[0]}")
            
                logger.info('Saving train and test datasets')
                os.makedirs(os.path.dirname(self.train_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.test_output_path), exist_ok=True)
                
                train_df.to_csv(self.train_output_path, index=False)             #splitted into train csv
                test_df.to_csv(self.test_output_path, index=False)                #splitted into test csv
                logger.info('Train and Test set is saved successfully')
                
            except Exception as e:
                logger.error(f'Error during data splitting => src/Data_preprocessing/Data_split/Data_split.py => run_data_split() => {e}')
                return CustomException('Error during data splitting', sys)
            
# Note: The data splitting is done before feature engineering to prevent data leakage.
# Next feature engineering (ipynb to py)