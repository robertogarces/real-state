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

df = import_raw_dataset()
features = read_yaml(f'{CONFIG_PATH}/features.yaml')

target = features['target'][0]

df = remove_duplicated_ids(df)
df = remove_price_outliers(df, lower_bound=2.5, upper_bound=97.5)

pctg_train = 0.8
n_train = int(len(df) * pctg_train)
train_idx = df.sample(n=n_train, random_state=42).index
train = df.loc[train_idx]
test = df.loc[~df.index.isin(train_idx)]

transform_price_log = FunctionTransformer(transform_price_log, validate=False)
transform_area_units = FunctionTransformer(transform_area_units, validate=False)
categorize_bedrooms = FunctionTransformer(categorize_bedrooms, validate=False)
categorize_bathrooms = FunctionTransformer(categorize_bathrooms, validate=False)
categorize_yearBuilt = FunctionTransformer(categorize_yearBuilt, validate=False)
remove_garageSpaces_outliers = FunctionTransformer(remove_garageSpaces_outliers, validate=False)
map_levels = FunctionTransformer(map_levels, validate=False)
process_homeType = FunctionTransformer(process_homeType, validate=False)
impute_hasGarage = FunctionTransformer(impute_hasGarage, validate=False)
city_median_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'city', 'statistic': 'median'})
city_mean_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'city', 'statistic': 'mean'})
county_median_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'county', 'statistic': 'median'})
county_mean_price = FunctionTransformer(calculate_statistic, validate=False, kw_args={'feature': 'county', 'statistic': 'mean'})
_5_knn_median_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':5, 'statistic':'median'})
_5_knn_mean_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':5, 'statistic':'mean'})
_25_knn_median_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':25, 'statistic':'median'})
_25_knn_mean_price = FunctionTransformer(knn_property_price, validate=False, kw_args={'n_neighbors':25, 'statistic':'mean'})
encoding = FunctionTransformer(encode_categorical_variables, validate=False)
drop_features = FunctionTransformer(drop_features, validate=False, kw_args={'target':target})

preprocessing_steps = [
    ('transform_price_log', transform_price_log),
    ('transform_area_units', transform_area_units),
    ('categorize_bedrooms', categorize_bedrooms),
    ('categorize_bathrooms', categorize_bathrooms),
    ('categorize_yearBuilt', categorize_yearBuilt),
    ('remove_garageSpaces_outliers', remove_garageSpaces_outliers),
    ('map_levels', map_levels),
    ('process_homeType', process_homeType),
    ('impute_hasGarage', impute_hasGarage),
    ('city_median_price', city_median_price),
    ('city_mean_price', city_mean_price),
    ('county_median_price', county_median_price),
    ('county_mean_price', county_mean_price),
    ('5_knn_median_price', _5_knn_median_price),
    ('5_knn_mean_price', _5_knn_mean_price),
    ('25_knn_median_price', _25_knn_median_price),
    ('25_knn_mean_price', _25_knn_mean_price),
    ('encoding', encoding),
    ('drop_features', drop_features)
]

preprocessing_pipeline = Pipeline(preprocessing_steps)
train_transformed = preprocessing_pipeline.fit_transform(train.copy())
save_pkl(preprocessing_pipeline, f'{ARTIFACTS_PATH}/preprocessing_pipeline.pkl')

train_transformed.to_parquet(f'{PROCESSED_DATA_PATH}/transformed_dataset.parquet', index=False)

model = train_lightgbm_model(train_transformed, target)
save_pkl(model, f'{ARTIFACTS_PATH}/model.pkl')
plot_feature_importance(model)

test_transformed = preprocessing_pipeline.transform(test.copy())
evaluate_test_set(test_transformed, model, target, decimals=3, log_inverse_transform=False)






