import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import os

from config.paths import RAW_DATA_PATH

def import_raw_dataset():

    filename = 'RealEstate_California.csv'
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(filepath)
    print(f'Successfully import {filepath}')

    return df


def remove_duplicated_ids(df):

    df.sort_values(['id', 'datePostedString'])
    df.drop_duplicates(subset="id", keep='first', inplace=True)

    return df

def remove_price_outliers(
        df, 
        lower_bound=2.5, 
        upper_bound=97.5
    ):

    df = df[df['price'] > 0]

    lower_bound_price = np.percentile(df['price'], lower_bound)
    upper_bound_price = np.percentile(df['price'], upper_bound)

    print(lower_bound_price, upper_bound_price)

    df = df[df['price'].between(lower_bound_price, upper_bound_price)]

    return df


def transform_price_log(df):

    df['price_log'] = np.log(df['price'])

    return df


def replace_zeros_with_nans(
        feature
    ):

    feature = np.where(
        feature==0,
        np.NAN,
        feature
        )

    return feature


def discretize_data(data, method='uniform', bins=5):
    """
    Discretize numerical data using different bucketing methods.

    Parameters:
    - data: pandas Series, the numerical data to be discretized.
    - method: str, the bucketing method. Options: 'uniform', 'quantile', 'kmeans'.
    - bins: int or sequence of scalars, the number of bins or the bin edges.

    Returns:
    - pandas Series, the discretized data.
    """
    if method == 'uniform':
        return pd.cut(data, bins=bins, labels=False, include_lowest=True)
    elif method == 'quantile':
        return pd.qcut(data, q=bins, labels=False, duplicates='drop')
    elif method == 'kmeans':
        data_reshaped = data.values.reshape(-1, 1)
        kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
        discretized_data = kbd.fit_transform(data_reshaped)
        return pd.Series(discretized_data.flatten(), index=data.index)
    else:
        raise ValueError("Invalid method. Supported methods: 'uniform', 'quantile', 'kmeans'.")


def transform_area_units(df):
    
    df['livingAreaMts'] = np.where(
        df['lotAreaUnits'] == 'sqft',
        df['livingArea'] * 0.092903,
        df['livingArea'] * 4046.86
        )

    df['livingAreaMts_log'] = np.log1p(df['livingArea'])

    return df


def categorize_bedrooms(df):

    bedroom_conds = [
        df['bedrooms']==0,
        df['bedrooms']>=5,
        df['bedrooms']==4,
        df['bedrooms']==3,
        df['bedrooms']<=2,
        df['bedrooms'].isna()
    ]

    bedroom_values = ['NaN', '>=5', '4', '3', '<=2', 'NaN']

    df['bedrooms_disc'] = np.select(bedroom_conds, bedroom_values)
    return df

def categorize_bathrooms(df):

    bathroom_conds = [
        df['bathrooms']>=4,
        df['bathrooms']==3,
        df['bathrooms']==2,
        df['bathrooms']==1,
        df['bathrooms']==0,
        df['bathrooms'].isna()
    ]

    bathroom_values = ['>=4', '3', '2', '1', 'NaN', 'NaN']

    df['bathrooms_disc'] = np.select(bathroom_conds, bathroom_values)

    return df

def categorize_yearBuilt(df):
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
    df['yearBuilt_disc'] = np.select(yearBuilt_conds, yearBuilt_values)

    return df


def remove_garageSpaces_outliers(df):

    df['garageSpaces'] = np.where(
        df['garageSpaces']>4,
        4,
        df['garageSpaces']
        )
    
    return df


def map_levels(df):

    mapper = {
      "0": "0",
      "One Story": "1",
      "Two Story": "2",
      "One": "1",  # Consider merging with "One Story"
      "Three Or More": ">=3",
      "Two": "2",  # Consider merging with "Two Story"
      "Multi/Split": ">=3",
      "One-Two": "1",  # Consider merging with "One" or "Two"
      "Three": ">=3",
      "Tri-Level": ">=3",
      "Three Or More-Multi/Split": ">=3",
      "Four": ">=3",
      "One Story-Three Or More": ">=3",
      "One Story-One": "1",  # Consider merging with "One Story"
      "Three or More Stories": ">=3",
      "Two-Multi/Split": ">=3",
      "Two Story-Two": "2",  # Consider merging with "Two Story"
      "Three or More Stories-Three Or More": ">=3",
      "Two Story-One": "2",  # Consider merging with "Two Story"
      "Multi/Split-Tri-Level": ">=3",
      "Other": "0",  # Keep "Other" category if desired
      "2": "2",  # Consider merging with "Two" or "Two Story"
      "Tri-Level-Two": ">=3",
      "Three or More Stories-One": ">=3",
      "Multi/Split-Three Or More": ">=3",
      "One-Multi/Split": "1",  # Consider merging with "One"
      "One-Two-Three Or More": ">=3",
      "Two Story-Three Or More": ">=3",
      "Three or More Stories-Two": ">=3",
      "One-Three Or More": ">=3",
      "1": "1",  # Consider merging with "One" or "One Story"
      "3": ">=3",
      "Multi/Split-One": "1",  # Consider merging with "One"
      "Five or More": ">=3",
      "Split Level": ">=3",
      "One-Two-Multi/Split": ">=3",
      "Three Or More-Split Level": ">=3",
      "Multi/Split-Two": ">=3",
      "Other-One": "1",  # Keep "Other" category if desired
      "Two-Three Or More": ">=3",
      "One Story-Two": "2",  # Consider merging with "Two Story"
      "4+": ">=3",
      "Tri-Level-Three Or More": ">=3",
      "Multi-Level": ">=3",
      "Three Or More-Two": ">=3",
      "Three or More Stories-One-Two": ">=3",
      "Two-Three Or More-Multi/Split": ">=3",
      "Two-One": "2",  # Consider merging with "Two" or "Two Story"
    }

    df['mapped_levels'] = [mapper.get(level, level) for level in df['levels']]
    mapping_levels = {'0': 0, '1': 1, '2': 2, '>=3': 3}
    df['mapped_levels'] = df['mapped_levels'].map(mapping_levels)
  
    return df


def process_homeType(df):

    df['homeType'] = np.where(df['homeType']=='APARTMENT', 'TOWNHOUSE', df['homeType'])

    return df


def impute_hasGarage(df):

    # Impute hasGarage
    df['hasGarage'] = np.where(
        (df['hasGarage']==0) & (df['garageSpaces']!=0),
        1,
        df['garageSpaces']
        )
    
    return df


def calculate_statistic(df, feature, statistic):
    df[f'{feature}_{statistic}_price'] = df.groupby(feature)['price'].transform(statistic)
    return df


def drop_features(df_, target):
    cols_to_use = [target,'id', 'city', 'county', 'lotAreaUnits', 'parking', 'garageSpaces', 'hasGarage', 'pool',
                   'spa', 'homeType', 'livingAreaMts_log', 'yearBuilt', 'mapped_bathrooms', 'city_median_price',
                   'city_mean_price', 'county_median_price', 'county_mean_price', 'bedrooms', 'levels',
                   '5_knn_mean_price', '5_knn_median_price', '25_knn_mean_price', '25_knn_median_price']

    # Filtrar las columnas que existen en el DataFrame
    cols_to_use = [col for col in cols_to_use if col in df_.columns]

    df_ = df_[cols_to_use]

    return df_


import pandas as pd
from sklearn.neighbors import BallTree

def knn_property_price(df_, n_neighbors=3, statistic='mean'):

    df_['latitude'] = df_['latitude'].astype(float)
    df_['longitude'] = df_['longitude'].astype(float)

    # Combina las coordenadas en una matriz
    coords = df_[['latitude', 'longitude']].values

    # Construye un árbol de bolas para buscar vecinos cercanos eficientemente
    tree = BallTree(coords, leaf_size=15, metric='haversine')

    # Para cada propiedad, encuentra los índices de las N propiedades más cercanas
    _, indices = tree.query(coords, k=n_neighbors + 1)

    # Calcula la estadística especificada de las propiedades cercanas para cada fila
    if statistic == 'mean':
        df_[f'{n_neighbors}_knn_{statistic}_price'] = [df_.iloc[idx]['price'][1:].astype(float).mean() for idx in indices]
    elif statistic == 'median':
        df_[f'{n_neighbors}_knn_{statistic}_price'] = [df_.iloc[idx]['price'][1:].astype(float).median() for idx in indices]
    elif statistic == 'std':
        df_[f'{n_neighbors}_knn_{statistic}_price'] = [df_.iloc[idx]['price'][1:].astype(float).std() for idx in indices]
    # Puedes agregar más opciones según tus necesidades

    return df_


def encode_categorical_variables(df):
  """
  Encodes categorical variables in a dataset.

  Args:
    df: The dataset.

  Returns:
    The encoded dataset.
  """

  # Get the categorical variables.
  categorical_variables = [
      var for var in df.columns if df[var].dtype.name == "object"
  ]

  # Encode the categorical variables.
  for var in categorical_variables:
    df[var] = df[var].astype("category")
    df[var] = df[var].cat.codes

  return df




