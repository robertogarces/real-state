import pandas as pd
import sys 
sys.path.append('../')
pd.set_option('display.max_columns', None)

from utils.processing import *
from utils.file_management import read_yaml, save_pkl, load_pkl
from utils.train import train_lightgbm_model
from utils.evaluation import mape_score
from utils.plotting import plot_feature_importance
from utils.evaluation import evaluate_test_set
from utils.optimizer import optimize_lightgbm_params

from config.paths import CONFIG_PATH, PROCESSED_DATA_PATH, ARTIFACTS_PATH
from config.config import USE_OPTUNA, N_TRIALS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import mlflow
import mlflow.sklearn

from metaflow import FlowSpec, step

class TrainModel(FlowSpec):

 #   mlflow.start_run()
    features = read_yaml(f'{CONFIG_PATH}/features.yaml')
    target = features['target'][0]
    preprocessing_pipeline = load_pkl(f'{ARTIFACTS_PATH}/preprocessing_pipeline.pkl')

    @step
    def start(self):

        print("Initializing model training pipeline")
        self.next(self.load_data)
        
    @step
    def load_data(self):

        self.train = import_preprocessed_train_dataset()
        self.test = import_test_dataset()

        self.next(self.hipermarameter_optimization)

    @step
    def hipermarameter_optimization(self):

        if USE_OPTUNA:

            self.best_params = optimize_lightgbm_params(
                self.train.drop(self.target, axis=1),
                self.train[self.target],
                n_trials=N_TRIALS
                )
            
        else:
            pass
        
        self.next(self.train_model)


    @step
    def train_model(self):

        if USE_OPTUNA:
            self.model = train_lightgbm_model(self.train, self.target, self.best_params)
        else:
            self.model = train_lightgbm_model(self.train, self.target)

        save_pkl(self.model, f'{ARTIFACTS_PATH}/model.pkl')
        plot_feature_importance(self.model)
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):

        test_transformed = self.preprocessing_pipeline.transform(self.test.copy())
        evaluate_test_set(test_transformed, self.model, self.target, decimals=3, log_inverse_transform=False)
        self.next(self.end)

    @step
    def end(self):

        print("Model trained successfully")

if __name__ == '__main__':
    TrainModel()


