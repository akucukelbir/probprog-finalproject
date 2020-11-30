
import predictors
import pandas as pd
import numpy as np
from constants import *

def get_accidents(df):
    """
    Returns [day,node] nd.array where arr[i][j] indicates
    accidents in location i at j day
    Returns accident array
    """

    #Categories for classifying the codes
    categorical = pd.Categorical(df['node'])
    codes = categorical.codes
    df['datetime'] = pd.to_datetime(df['datetime'])
    num_days = (df['datetime'].iloc[-1] - FIRST_DAY).days + 1
    num_nodes = len(categorical.categories)

    data_arr = np.zeros((num_nodes, num_days))
    for elem, i in zip(df.itertuples(),range(len(df))):
        data_arr[codes[i]][(elem.datetime - FIRST_DAY).days] += 1

    category_mapping = {}
    for node, idx in zip(categorical, categorical.codes):
        category_mapping[node] = idx

    return category_mapping, data_arr

def get_data():
    weather_df = pd.read_csv("../data/weather/processed/data.csv")
    intersection_df = pd.read_csv("../data/intersection/processed/data.csv")
    accident_df = pd.read_csv("../data/accident/processed/manhattan.csv")
    accident_df = accident_df[accident_df['node'].isin(list(intersection_df['nodes'].unique()))]
    accident_df['datetime'] = pd.to_datetime(accident_df['datetime'])
    accident_df = accident_df[accident_df['datetime'] >= FIRST_DAY]
    categorical_mapping, accidents = get_accidents(accident_df)
    accidents = accidents[:NUM_DAYS]
    pred = predictors.get_predictors(weather_df, intersection_df, categorical_mapping)
    return accidents, pred
