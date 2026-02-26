import os
import yaml
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Utils.Logging.logger import logging
from Utils.Exception.exception import CustomException

class TrainingPipeline:
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
    
        self.train_path = self.config['data']['train_preprocessed_path']
        self.test_path = self.config['data']['test_preprocessed_path']
        
        self.model_params = self.config['model']['params']
        
        self.save_model_apth = 'Artifacts/Model/model_LGBM.pkl'
        
        
    def _load_config(self):
        try:
            with open(self.config_path,'r') as f:
                logging.info('Loging configuration files...')
                return yaml.safe_load(f)
        except Exception as e:
            logging.error('Failed to load the config files, Check the function of _load_config')
            raise CustomException(f'Failed to load the configuration files {e}', sys)
   
        
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info('Loding the data')
            
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            logging.info('Data loaded successfully..')
            
            return train_df, test_df
        
        except Exception as e:
            logging.error('Error loading the data')
            raise CustomException(f'Error due to loading data {e}', sys) 
    
    
    def _select_features(self, df: pd.DataFrame):
        try:
            logging.info('Selecting featured for training')
            
            drop_columns = [
                'engine_id',
                "fault_label",
                "fault_type",
                "fault_target",
                "RUL",
            ]
            
            x = df.drop(columns=drop_columns, errors='ignore')
            y = df['RUL']
            
            logging.info('Selected the features')
            
            return x, y
    
        except Exception as e:
            logging.error('Error due to selecting the features foe the training in "training_pipeline.py" ')
            raise CustomException(f'Error due to selection features for training {e}', sys)


    def _train_model(self, x_train, y_train):
        try:
            logging.info('Training LightGBM model...')
            
            model = LGBMRegressor(**self.model_params)
            model.fit(x_train, y_train)
            
            logging.info("Model training completed.")
            return model
        
        except Exception as e:
            logging.error('Error occured in model traininig')
            raise CustomException(f'Error due to model training {e}', sys)
        
    def _evaluate_model(self, model, x_test, y_test):
        try:
            logging.info("Evaluating model performance...")
            
            preds = model.predict(x_test)
            
            
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mse)

            logging.info(f"MSE: {mse:.3f}")
            logging.info(f"MAE : {mae:.3f}")
            logging.info(f"R2  : {r2:.3f}")
            logging.info(f"RMSE : {rmse:.3f}")

            return mse, mae, r2, rmse
        
        except Exception as e:
            logging.error('Error due to evaluating model....')
            raise CustomException(f'Error while evaluating model {e}', sys)
        
    def _save_model(self, model):
        try:
            
            os.makedirs(os.path.dirname(self.save_model_apth), exist_ok=True)
            joblib.dump(model, self.save_model_apth)
            
            logging.info(f'Model saved at : {self.save_model_apth}')
            
        except Exception as e:
            logging.error('Error occured while saving the model')
            raise CustomException(f'Error due to saving the model {e}', sys)
        
    
    def run_training_pipeline(self):
        try:
            logging.info('Training Pipeline begins...')
            
            train_df, test_df = self._load_data()
            
            x_train, y_train = self._select_features(train_df)
            x_test, y_test = self._select_features(test_df)
            
            logging.info(f"Training shape: {x_train.shape}")
            logging.info(f"Testing shape: {x_test.shape}")
            
            model = self._train_model(x_train, y_train)
            
            self._evaluate_model(model, x_test, y_test)
            
            self._save_model(model)
            
            logging.info("Training pipeline completed successfully.")

        except Exception as e:
            logging.error("Error in training pipeline.")
            raise CustomException(e, sys)
            
        
    