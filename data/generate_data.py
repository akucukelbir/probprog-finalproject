#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import osmnx as ox
from pandarallel import pandarallel


pandarallel.initialize()
newyork_graph = ox.io.load_graphml('newyork.graphml')

# These parameters were determined by looking at a map
LOCATION = [40.7128, -74.0060]
TIME = [pd.to_datetime("2012-07-01"), pd.to_datetime("2020-10-31")]
LOWER_LEFT = [LOCATION[0] - 0.3, LOCATION[1] - 0.3]
UPPER_RIGHT = [LOCATION[0] + 0.3, LOCATION[1] + 0.4]
FILEPATH = 'reduced_data.csv'

#NUMBER of ways to divide the grid by 
NUM_DIVISIONS = [10, 100, 500, 1000]

df = pd.read_csv(FILEPATH, index_col=0)

def aggregate_data(df, lower_left, upper_right, n):
    """
    lower_left: lower left lat and long of grid of grids
    upper_right: upper left lat and long of grid of girds
    number of divisions
    """
    df_copy = pd.DataFrame.copy(df)
    
    lat_steps = np.linspace(lower_left[0], upper_right[0], n+1)
    lon_steps = np.linspace(lower_left[1], upper_right[1], n+1)
    
    xleft = lower_left[1]
    xright = upper_right[1]
    
    ybot = lower_left[0]
    yup = upper_right[0]
    
    xstride = (xright - xleft)/n
    ystride = (yup - ybot)/n
    
    df_copy['x_box'] = np.floor((df_copy['longitude'] - xleft)/xstride).astype(np.int)
    df_copy['y_box'] = np.floor((df_copy['latitude'] - ybot)/ystride).astype(np.int)
    print(len(df_copy.groupby(['x_box', 'y_box']).size().reset_index(name='counts')))
    df_copy['repr_lat'] = df_copy['y_box']* ystride + ystride/2 + ybot
    df_copy['repr_long'] = df_copy['x_box']* xstride + xstride/2 + xleft

    return df_copy


def populate_nearest_node(row):
    point = (row['latitude'], row['longitude'])
    return ox.distance.get_nearest_node(newyork_graph, point)

df['node'] = df.parallel_apply(populate_nearest_node, axis=1)
df.to_csv("reduced_node_data.csv", index=False)


#for n in range(1, 100):
#    data = aggregate_data(df, LOWER_LEFT, UPPER_RIGHT, 50*n)
#    for row in data:
#        print(row)
#for n in NUM_DIVISIONS:
#    data = aggregate_data(df, LOWER_LEFT, UPPER_RIGHT, n)
#    data.to_csv("aggregated_data_{}_divisions.csv".format(n),index=False)
#    print("Done generating data with {} divisions".format(n))


# In[ ]:




