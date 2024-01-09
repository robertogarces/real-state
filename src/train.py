import pandas as pd
import sys

import sys
sys.path.append('../')

from config.paths import CONFIG_PATH, ARTIFACTS_PATH, ROOT_PATH
from config.config import (
    TARGET,
    USE_OPTUNA,
    USE_OPTIMIZED_PARAMS,
    N_TRIALS,
    EVALUATION_METRIC_DECIMALS,
    LOG_INVERSE_TRANSFORM,
    DEFAULT_LGBM_PARAMS,
    SAVE_OPTIMIZED_PARAMS,
)

from utils.processing import *
from utils.file_management import read_yaml, save_pkl, load_pkl, save_json, read_json
from utils.train import train_lightgbm_model
from utils.plotting import plot_feature_importance
from utils.evaluation import model_evaluation
from utils.optimizer import optimize_lightgbm_params

import mlflow
import mlflow.sklearn

from metaflow import FlowSpec, step


class TrainModel(FlowSpec):
    mlflow.start_run()
    features = read_yaml(f"{CONFIG_PATH}/features.yaml")
    target = features["target"][TARGET]
    preprocessing_pipeline = load_pkl(f"{ARTIFACTS_PATH}/preprocessing_pipeline.pkl")

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
        best_params_path = f"{CONFIG_PATH}/best_model_params.json"

        if USE_OPTUNA:
            print("Optimizing hyper parameters using Optuna")
            self.params = optimize_lightgbm_params(
                self.train.drop(self.target, axis=1),
                self.train[self.target],
                n_trials=N_TRIALS,
            )

            print("Best hyper parameters")
            for key, value in self.params.items():
                print(f" • {key}: {value}")

            print(self.params)

            if SAVE_OPTIMIZED_PARAMS:
                save_json(self.params, best_params_path)
                print(
                    f"Optimized hyper parameters saved successfully in: {best_params_path}"
                )

        else:
            self.params = DEFAULT_LGBM_PARAMS

        if USE_OPTIMIZED_PARAMS:
            self.params = read_json(best_params_path)

        self.next(self.train_model)

    @step
    def train_model(self):
        self.model = train_lightgbm_model(self.train, self.target, self.params)

        save_pkl(self.model, f"{ARTIFACTS_PATH}/model.pkl")
        plot_feature_importance(self.model)
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        test_transformed = self.preprocessing_pipeline.transform(self.test.copy())
        self.r2, self.mape, self.rmse, self.median_absolute_error = model_evaluation(
            test_transformed,
            self.model,
            self.target,
            decimals=EVALUATION_METRIC_DECIMALS,
            log_inverse_transform=LOG_INVERSE_TRANSFORM,
        )

        self.next(self.end)

    @step
    def end(self):
        mlflow.log_param("target", TARGET)
        mlflow.log_param("use_optuna", USE_OPTUNA)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("evaluation_metric_decimals", EVALUATION_METRIC_DECIMALS)
        mlflow.log_param("log_inverse_transform", LOG_INVERSE_TRANSFORM)
        mlflow.log_param("model_params", self.params)
        mlflow.log_param("use_optuna", USE_OPTUNA)

        mlflow.log_metric("r2", self.r2)
        mlflow.log_metric("mape", self.mape)
        mlflow.log_metric("rmse", self.rmse)
        mlflow.log_metric("median_absolute_error", self.median_absolute_error)

        mlflow.sklearn.log_model(self.model, "model")

        mlflow.end_run()

        print("Model trained successfully")


if __name__ == "__main__":
    TrainModel()
