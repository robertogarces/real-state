import pandas as pd
import numpy as np

import sys

sys.path.append('../')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.3f}'.format)

from config.paths import RAW_DATA_PATH, CONFIG_PATH, PROCESSED_DATA_PATH
from utils.file_management import read_yaml
from utils.processing import discretize_data, remove_outliers, replace_zeros_with_nans

df = pd.read_csv(f'{RAW_DATA_PATH}/RealEstate_California.csv').drop(columns=["Unnamed: 0"])
features = read_yaml(f'{CONFIG_PATH}/features.yaml')
target = features['target']

df.sort_values(['id', 'datePostedString'])
df.drop_duplicates(subset="id", keep='first', inplace=True)

columns_to_drop = features['constant_columns'] + features['unique_columns'] + features['heavily_imbalanced_columns'] + features['non_important_columns'] + features['data_leakage_columns']
df.drop(columns=columns_to_drop, inplace=True)

df = df[df['price'] > 0]

price_perc01 = np.percentile(df['price'], 2.5)
price_perc99 = np.percentile(df['price'], 97.5)

print(price_perc01, price_perc99)

df = df[df['price'].between(price_perc01, price_perc99)]

df['livingAreaMts'] = np.where(
    df['lotAreaUnits'] == 'sqft',
    df['livingArea'] * 0.092903,
    df['livingArea'] * 4046.86
    )

bedroom_conds = [
    df['bedrooms']==0,
    df['bedrooms']>=5,
    df['bedrooms']==4,
    df['bedrooms']==3,
    df['bedrooms']<=2,
    df['bedrooms'].isna()
]

bedroom_values = ['NaN', '>=5', '4', '3', '<=2', 'NaN']

df['bedrooms'] = np.select(bedroom_conds, bedroom_values)


bathroom_conds = [
    df['bathrooms']>=4,
    df['bathrooms']==3,
    df['bathrooms']==2,
    df['bathrooms']==1,
    df['bathrooms']==0,
    df['bathrooms'].isna()
]

bathroom_values = ['>=4', '3', '2', '1', 'NaN', 'NaN']

df['bathrooms'] = np.select(bathroom_conds, bathroom_values)

print(df['bathrooms'].value_counts(normalize=True))


yearBuilt_conds = [
    df['yearBuilt']==9999,
    df['yearBuilt']==0,
    df['yearBuilt']<1925,
    df['yearBuilt']<=1950,
    df['yearBuilt']<=1975,
    df['yearBuilt']<=2000,
    df['yearBuilt']<=2010,
    df['yearBuilt']>2010,
    df['yearBuilt'].isna()
]

yearBuilt_values = ['NaN', 'NaN', '<1925', '1925-1950', '1950-1975', '1975-2000', '2000-2010', '>2010', 'NaN']

df['yearBuilt'] = np.select(yearBuilt_conds, yearBuilt_values)


living_area = np.log1p(df['livingArea'])

df['price'] = np.log(df['price']).hist()




















