import predictors
import pandas as pd
import numpy as np

def get_accidents():
    """
    Returns [day,node] nd.array where arr[i][j] indicates
    accidents in location i at j day
    Returns accident array
    """
    
    #Categories for classifying the codes
    categorical = pd.Categorical(df['node'])
    codes = categorical.codes

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
    categorical_mapping, accidents = get_accidents()
    predictors = get_predictors(categorical_mapping)
    return accidents, predictors
