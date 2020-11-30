import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geom
import data_utils as utils
import time
import reverse_geocoder as rg
import importlib
import osmnx as ox
import re
import math


def process_aadt():
    aadt_complete = pd.read_csv("intersection/raw/aadt.csv", index_col=False)
    aadt_complete = aadt_complete.drop_duplicates().dropna(subset=["Count"])

    aadt_2019_map = pd.read_csv(
        "intersection/raw/aadt_2019_map.csv", index_col=0
    )
    aadt_2019_map = aadt_2019_map.rename(
        columns={
            "RoadwayName": "Road Name",
            "BeginDescription": "Beginning Description",
            "EndDescription": "Ending Description",
        }
    )

    # We only care about the last 6 years
    years = aadt_complete.Year.unique()
    years_aadt = []

    for year in years:
        years_aadt.append(aadt_complete[:][aadt_complete.Year == year])
    years = years[:6]
    years_aadt = years_aadt[:6]

    # Process each individual year dataframes
    for i, year in enumerate(years):
        years_aadt[i] = years_aadt[i].rename(
            columns={"Count": "Count_" + str(year)}
        )
        columns_to_drop = ["Year"]
        if i > 0:
            columns_to_drop = [
                "Ramp",
                "Bridge",
                "Railroad Crossing",
                "Year",
                "Station ID",
                "Municipality",
                "One Way",
                "Functional Class",
                "Length",
                "County",
                "Signing",
                "State Route",
                "County Road",
            ]
        years_aadt[i] = years_aadt[i].drop(columns_to_drop, axis=1)

    # Join the dataframes
    joined_aadt = years_aadt[0]
    for year_aadt in years_aadt[1:]:
        joined_aadt = pd.merge(
            year_aadt.drop_duplicates(),
            joined_aadt,
            how="outer",
            on=["Road Name", "Beginning Description", "Ending Description"],
        )

    joined_aadt = pd.merge(
        joined_aadt,
        aadt_2019_map,
        how="inner",
        on=["Road Name", "Beginning Description", "Ending Description"],
    )
    joined_aadt.to_csv(
        "intersection/intermediate/joined_aadt.csv", index=False
    )


def transform(x):
    points = []
    j = 0
    point = []
    for i in range(len(x)):
        if j == 0:
            point = [float(x[i])]
            j += 1
        elif j == 1:
            point.append(float(x[i]))
            points.append(point)
            j = 0
    return np.array(points)


def find_min_segment(point, segments, cutoff=True):
    """
    returns index of closest segment.
    Assumes point is np.array([x,y])
    Segments is an array of road semgments each item in
    the list represets one road.
    """

    def return_min(segment):
        return np.nanmin(utils.lineseg_dists(point, segment[:-1], segment[1:]))

    # We are looking for all roads closer than 0.01 km.
    # A unit in long-lat coordinates is equal to about 111 km
    distances = np.array([return_min(segment) for segment in segments])
    if cutoff:
        return (distances < (0.01 / 111)).nonzero()[0]
    else:
        return np.nanargmin(distances)


def match_road_to_node():
    data = pd.read_csv("intersection/intermediate/joined_aadt.csv")
    manhattan_accidents = pd.read_csv("accident/processed/manhattan.csv")
    unique_nodes, index_unique = np.unique(
        manhattan_accidents[["node"]].to_numpy(), return_index=True
    )
    graph_file = "reference/newyork.graphml"
    graph = ox.io.load_graphml(graph_file)
    unique_x = []
    unique_y = []
    num_connect = {}
    for node in unique_nodes:
        num_connect[node] = len(graph[node])
        unique_x.append(graph.nodes[node]["x"])
        unique_y.append(graph.nodes[node]["y"])
    unique_x = np.expand_dims(np.array(unique_x), 1)
    unique_y = np.expand_dims(np.array(unique_y), 1)
    unique_points = np.concatenate((unique_x, unique_y), axis=1)

    manhattan_roads = data[data["County"].apply(lambda x: x == "New York")]
    p = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
    xy = manhattan_roads["geometry"].apply(lambda x: p.findall(x))
    segments = xy.apply(transform)
    segments = segments.to_numpy()

    minim = []
    nodes_roads_data = {}

    cols = []
    for n in range(4, 10):
        cols.append("Count_201" + str(n))

    data["Count_mean"] = data[cols].mean(axis=1)
    cols.append("Count_mean")
    display(data.head())
    for col in cols:
        nodes_roads_data[col] = []

    for i in range(len(unique_points)):
        if i % 100 == 0:
            print(i)
        road_ids = list(
            find_min_segment(unique_points[i], segments, cutoff=True)
        )
        for col in cols:
            max_AADT = data.iloc[road_ids].dropna(subset=[col])[col].max()
            nodes_roads_data[col].append(max_AADT)

        minim.append(",".join(map(str, road_ids)))

    num_connect_arr = []
    for node in unique_nodes:
        num_connect_arr.append(num_connect[node])

    nodes_roads_data["nodes"] = unique_nodes
    nodes_roads_data["num_connect"] = num_connect_arr
    nodes_roads_data["roads"] = minim

    intersections = pd.DataFrame(data=nodes_roads_data)
    intersections = intersections.dropna(subset=cols, how="all")
    intersections.to_csv("intersection/processed/data.csv")
