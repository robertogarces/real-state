import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import os

from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH

def import_raw_dataset():

    filename = 'RealEstate_California.csv'
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(filepath)
    print(f'Successfully import {filepath}')

    return df


def import_preprocessed_train_dataset():

    filename = 'transformed_dataset.parquet'
    filepath = os.path.join(PROCESSED_DATA_PATH, filename)
    df = pd.read_parquet(filepath)
    print(f'Successfully import {filepath}')

    return df

def import_test_dataset():


    filename = 'test_dataset.parquet'
    filepath = os.path.join(PROCESSED_DATA_PATH, filename)
    df = pd.read_parquet(filepath)
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

    cols_to_drop = [
        'Unnamed: 0', # ID
        'id', # ID
        'countyId', # Correlated with county_median_price and county_mean_price
        'cityId', # Correlated to city_median_price and city_mean_price
        'country', # Correlated to countyId
        'datePostedString', # Heavily imbalanced and almost all of the dates are from a single period
        'is_bankOwned', # Heavily imbalanced
        'is_forAuction', # Heavily imbalanced
        'event', # We are just keeping the first event (property first listing)
        'time', # Nothing relevant to do with this right now
        'pricePerSquareFoot', # Data leakage
        'city', # Correlated to cityId
        'stateId', # All of the properties are from the same state (California)
        'state', # All of the properties are from the same state (California)
        'streetAddress', # ID
        'zipcode', # Correlated to countyId and CityId
        'hasBadGeocode', # Heavily imbalanced
        'description', # Maybe there's something to do with this feature, but it's out of the scoop of this project
        'currency', # All of the currency are the same (USD)
        'livingAreaValue', # Data leakage
        'livingArea', # We are using livingAreaMts_log
        'livingAreaMts', # We are using livingAreaMts_log
        'buildingArea', # Heavily imbalanced and correlated to livingArea
        'hasGarage', # Correlated to garageSpaces
        'isNewConstruction', # Heavily imbalanced
        'hasPetsAllowed', # Heavily imbalanced
        'bedrooms', # We're using the mapped version of this feature
        'bathrooms', # We're using the mapped version of this feature
    ]

    # Filtra las columnas existentes en el DataFrame
    cols_to_drop = [col for col in cols_to_drop if col in df_.columns]

    # Elimina las columnas del DataFrame
    df_.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Mantén el target seleccionado y elimina el otro
    if target == 'price':
        other_target = 'price_log'
    elif target == 'price_log':
        other_target = 'price'
    else:
        raise ValueError("El valor del target debe ser 'price' o 'price_log'.")

    if target in df_.columns and other_target in df_.columns:
        df_.drop(columns=other_target, inplace=True)

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


import pandas as pd

def remove_highly_correlated_features(dataset, target, threshold=0.8):
    # Separate features and target variable
    X = dataset.drop(columns=[target])

    # Calculate correlation matrix
    correlation_matrix = X.corr()

    # Find pairs of highly correlated features
    high_corr_pairs = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >  threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                high_corr_pairs.add((colname_i, colname_j))

    # Remove features involved in highly correlated pairs
    features_to_remove = set()
    for pair in high_corr_pairs:
        features_to_remove.add(pair[1])

    reduced_dataset = dataset.drop(columns=features_to_remove)

    return reduced_dataset


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def cluster_model_feature(dataset, features, max_clusters=5):

    X = dataset[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    for i in range(2, max_clusters + 1):  # Empezamos desde 2 clusters
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette_avg)

    # Find the optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 porque empezamos desde 2 clusters
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Build and return the clustering model with the optimal number of clusters
    clustering_model = KMeans(n_clusters=optimal_clusters, random_state=42)
    clustering_model.fit(X_scaled)

    return clustering_model



def apply_cluster_model_feature(dataset, clustering_model, features):
    # Escalar las features del dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataset[features])

    # Obtener las predicciones de cluster
    cluster_predictions = clustering_model.predict(X_scaled)

    # Agregar las predicciones como una nueva columna al dataset
    dataset_with_clusters = dataset.copy()
    dataset_with_clusters['Cluster'] = cluster_predictions

    return dataset_with_clusters