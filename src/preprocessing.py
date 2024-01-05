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

from config.paths import CONFIG_PATH, PROCESSED_DATA_PATH, ARTIFACTS_PATH
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import mlflow
import mlflow.sklearn

from metaflow import FlowSpec, step

class Preprocessing(FlowSpec):

 #   mlflow.start_run()
    features = read_yaml(f'{CONFIG_PATH}/features.yaml')
    target = features['target'][0]

    @step
    def start(self):

        print("Initializing preprocessing pipeline")
        self.next(self.load_data)

    @step
    def load_data(self):

        self.df = import_raw_dataset()
        self.next(self.preprocessing)

    @step
    def preprocessing(self):

        self.df = remove_duplicated_ids(self.df)
        self.df = remove_price_outliers(self.df, lower_bound=2.5, upper_bound=97.5)
        self.next(self.split_datasets)

    @step
    def split_datasets(self):

        pctg_train = 0.8
        n_train = int(len(self.df) * pctg_train)
        train_idx = self.df.sample(n=n_train, random_state=42).index
        self.train = self.df.loc[train_idx]
        self.test = self.df.loc[~self.df.index.isin(train_idx)]

        self.test.to_parquet(f'{PROCESSED_DATA_PATH}/test_dataset.parquet', index=False)
        self.next(self.transform_functions)

    @step
    def transform_functions(self):

        self.transform_price_log = FunctionTransformer(transform_price_log, validate=False)
        self.transform_area_units = FunctionTransformer(transform_area_units, validate=False)
        self.categorize_bedrooms = FunctionTransformer(categorize_bedrooms, validate=False)
        self.categorize_bathrooms = FunctionTransformer(categorize_bathrooms, validate=False)
        self.categorize_yearBuilt = FunctionTransformer(categorize_yearBuilt, validate=False)
        self.remove_garageSpaces_outliers = FunctionTransformer(remove_garageSpaces_outliers, validate=False)
        self.map_levels = FunctionTransformer(map_levels, validate=False)
        self.process_homeType = FunctionTransformer(process_homeType, validate=False)
        self.impute_hasGarage = FunctionTransformer(impute_hasGarage, validate=False)
        self.city_median_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'city', 'statistic': 'median'})
        self.city_mean_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'city', 'statistic': 'mean'})
        self.county_median_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'county', 'statistic': 'median'})
        self.county_mean_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'county', 'statistic': 'mean'})
        self._5_knn_median_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':5, 'statistic':'median'})
        self._5_knn_mean_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':5, 'statistic':'mean'})
        self._25_knn_median_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':25, 'statistic':'median'})
        self._25_knn_mean_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':25, 'statistic':'mean'})
        self.encoding = FunctionTransformer(encode_categorical_variables, validate=False)
        self.drop_features = FunctionTransformer(drop_features, validate=False, kw_args={'target': self.target})

        self.next(self.define_preprocessing_pipeline)

    @step   
    def define_preprocessing_pipeline(self):

        self.preprocessing_steps = [
            ('transform_price_log', self.transform_price_log),
            ('transform_area_units', self.transform_area_units),
            ('categorize_bedrooms', self.categorize_bedrooms),
            ('categorize_bathrooms', self.categorize_bathrooms),
            ('categorize_yearBuilt', self.categorize_yearBuilt),
            ('remove_garageSpaces_outliers', self.remove_garageSpaces_outliers),
            ('map_levels', self.map_levels),
            ('process_homeType', self.process_homeType),
            ('impute_hasGarage', self.impute_hasGarage),
            ('city_median_price', self.city_median_price),
            ('city_mean_price', self.city_mean_price),
            ('county_median_price', self.county_median_price),
            ('county_mean_price', self.county_mean_price),
            ('5_knn_median_price', self._5_knn_median_price),
            ('5_knn_mean_price', self._5_knn_mean_price),
            ('25_knn_median_price', self._25_knn_median_price),
            ('25_knn_mean_price', self._25_knn_mean_price),
            ('encoding', self.encoding),
            ('drop_features', self.drop_features)
        ]
        self.next(self.fit_preprocessing_pipeline)

    @step
    def fit_preprocessing_pipeline(self):

        preprocessing_pipeline = Pipeline(self.preprocessing_steps)
        self.train_transformed = preprocessing_pipeline.fit_transform(self.train.copy())
        save_pkl(preprocessing_pipeline, f'{ARTIFACTS_PATH}/preprocessing_pipeline.pkl')

        self.train_transformed.to_parquet(f'{PROCESSED_DATA_PATH}/transformed_dataset.parquet', index=False)

        self.next(self.end)

    @step
    def end(self):

        print("Preprocessing pipeline ended successfully")

if __name__ == '__main__':
    Preprocessing()



