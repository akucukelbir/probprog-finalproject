import numpy as np
import pandas as pd
import osmnx as ox
from pandarallel import pandarallel
from sklearn.neighbors import KDTree
import shapely.geometry as geom
import data_utils as utils
import time 
import reverse_geocoder as rg
import importlib
import re
import geopandas as gpd
import math

def populate_nearest_node(row):
    global tree, points_ID, points
    point = (row['longitude'], row['latitude'])
    dist, ind = tree.query([point], k=1)
    osmid = points_ID[tuple(points[ind[0][0]])]
    return osmid

def filter_manhattan():
    nodesdf = pd.read_csv("accident/intermediate/with_node.csv")
    boro = gpd.read_file('reference/borough_boundaries.geojson')
    manhattan_poly = boro['geometry'].iloc[2]
    points = nodesdf[['longitude','latitude']].to_numpy()
    coordinates = [geom.Point(point) for point in points]
    inside_manhattan = [point.within(manhattan_poly) for point in coordinates]
    manhattan_accidents = nodesdf[inside_manhattan]
    manhattan_accidents.to_csv("accident/processed/manhattan.csv") #manhattan_accidents_node_data.csv

def clean():
    file_path = 'accident/raw/crashes.csv'
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime((df["CRASH DATE"] + " " + df["CRASH TIME"]))
    df.sort_values(by=["datetime"], inplace=True, ascending=False)
    df = df.dropna(subset=['LATITUDE','LONGITUDE'])
    df_clean = df[['datetime','LONGITUDE','LATITUDE']]
    df_clean.columns = ['datetime', 'longitude','latitude']
    df_clean.reset_index(drop=True, inplace=True)
    df_clean = df_clean.reindex(index=df_clean.index[::-1])
    df_clean.reset_index(drop=True, inplace=True)
    df_clean.to_csv('accident/intermediate/clean.csv',index=True)

def add_nearest_node():
    global tree, points_ID, points
    pandarallel.initialize()
    newyork_graph = ox.io.load_graphml('reference/newyork.graphml')

    df = pd.read_csv("accident/intermediate/clean.csv", index_col=0) #reduced_data.csv

    #Load nodes from the map into K-d tree and dictionary
    points = np.zeros((len(newyork_graph.nodes),2))
    points_ID = {}
    for idx, node in enumerate(newyork_graph.nodes):
        points[idx][0] = newyork_graph.nodes[node]['x']
        points[idx][1] = newyork_graph.nodes[node]['y']
        points_ID[tuple(points[idx])] = node
    tree = KDTree(points, leaf_size=1)
    df['node'] = df.parallel_apply(populate_nearest_node, axis=1)
    df.to_csv("accident/intermediate/with_node.csv", index=False) #accident_node_data.csv
