from folium.plugins import FastMarkerCluster, HeatMap
import folium

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import datetime


FIRST_DAY = datetime.datetime(2014,1,1)

accident_filename = '../data/accident/processed/manhattan.csv'
node_filename = '../data/intersection/processed/data.csv'


data = pd.read_csv(accident_filename)
node_data = pd.read_csv(node_filename)

#Only consider accident with node that have corresponding AADT
data = data[data['node'].isin(list(node_data['nodes'].unique()))]

#Only consider accidents after 2014
data['datetime'] = pd.to_datetime(data['datetime'])
data = data[data['datetime'] >= FIRST_DAY]
data_mat = data.to_numpy()



def make_heat_map():
    m = folium.Map(location=[40.7, -74.05], zoom_start=11)
    subset = data[['latitude','longitude']][:].values.tolist()
    m.add_child(HeatMap(subset, radius = 7.5))
    return m



def make_time_series(): 
    time_accidents = np.sum(data_mat, axis = 0)
    smooth_accidents = signal.savgol_filter(time_accidents,61, 3)
    time = list(range(len(time_accidents)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=time_accidents, mode='lines',name='Raw accidents'))
    fig.add_trace(go.Scatter(x=time,y=smooth_accidents,mode='lines',name='Smooth accidents'))
    fig.show()
