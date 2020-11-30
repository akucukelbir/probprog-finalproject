from constants import *
import pandas as pd
import numpy as np

def get_predictors(weather_df, intersection_df, categorical_mapping):
    first_week = np.array([0,0,0,1,1,0,0])
    predictors = np.zeros((len(categorical_mapping.keys()), NUM_DAYS, 8))
    predictors[:, :, 0] = 1
    intersection_df = intersection_df.replace({'nodes': categorical_mapping})
    for i in range(len(intersection_df)):
        if intersection_df['nodes'][i] >= 0 and intersection_df['nodes'][i] < predictors.shape[0]:
            predictors[intersection_df['nodes'][i], :, 1] = np.log(intersection_df['Count_mean'][i])
            predictors[intersection_df['nodes'][i], :, 2] = intersection_df['num_connect'][i] >= 3
    weather_df['datetime'] = pd.to_datetime(weather_df['DATE'])
    for elem in weather_df.itertuples():
        idx = (elem.datetime - FIRST_DAY).days
        if idx >= 0 and idx < predictors.shape[1]:
            predictors[:, idx, 3:7] = [elem.AWND, elem.PRCP, elem.SNWD, elem.TAVG]
    is_week = np.array([first_week[i%7] for i in range(NUM_DAYS)])
    predictors[:,:,7] = is_week
    return normalize(predictors)


def normalize(predictors):
    """
    Normalizes every variable using max 
    """
    normalization_constant =  np.max(np.max(predictors, axis=1), axis =0)
    return predictors/normalization_constant

def get_some_predictors(predictors, predictor_labels, categorical_mapping):
    pred_dict = {
            'aadt': 1,
            'is_intersection': 2,
            'wind': 3,
            'precipitation': 4,
            'snow_depth': 5,
            'temperature': 6,
            'weekend': 7
    }
    pred_idx = [0]
    for label in predictor_labels:
        pred_idx.append(pred_dict[label])
    if not predictors:
        predictors = get_predictors(categorical_mapping)
    return predictors[:, :, pred_idx]
