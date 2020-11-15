#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import osmnx as ox
from pandarallel import pandarallel
from sklearn.neighbors import KDTree

FILEPATH = 'reduced_data.csv'

pandarallel.initialize()
newyork_graph = ox.io.load_graphml('newyork.graphml')

df = pd.read_csv(FILEPATH, index_col=0)

#Load nodes from the map into K-d tree and dictionary
points = np.zeros((len(newyork_graph.nodes),2))
points_ID = {}
for idx, node in enumerate(newyork_graph.nodes):
    points[idx][0] = newyork_graph.nodes[node]['x']
    points[idx][1] = newyork_graph.nodes[node]['y']
    points_ID[tuple(points[idx])] = node
tree = KDTree(points, leaf_size=2)


def populate_nearest_node(row):
    point = (row['longitude'], row['latitude'])
    dist, ind = tree.query([point], k=1)
    osmid = points_ID[tuple(points[ind[0][0]])]
    return osmid

df['node'] = df.parallel_apply(populate_nearest_node, axis=1)
df.to_csv("accident_node_data.csv", index=False)
