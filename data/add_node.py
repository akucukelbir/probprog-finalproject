#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import osmnx as ox
from pandarallel import pandarallel


pandarallel.initialize()
newyork_graph = ox.io.load_graphml('newyork.graphml')

def populate_nearest_node(row):
    point = (row['latitude'], row['longitude'])
    return ox.distance.get_nearest_node(newyork_graph, point)

df['node'] = df.parallel_apply(populate_nearest_node, axis=1)
df.to_csv("reduced_node_data.csv", index=False)
