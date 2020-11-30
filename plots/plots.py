import sys

sys.path.insert(1, "../")

from folium.plugins import FastMarkerCluster, HeatMap
import folium

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import scipy

import datetime
import constants

accident_filename = "../data/accident/processed/manhattan.csv"
node_filename = "../data/intersection/processed/data.csv"


data = pd.read_csv(accident_filename)
node_data = pd.read_csv(node_filename)

# Only consider accident with node that have corresponding AADT
data = data[data["node"].isin(list(node_data["nodes"].unique()))]

# Only consider accidents after 2014
data["datetime"] = pd.to_datetime(data["datetime"])
data = data[data["datetime"] >= constants.FIRST_DAY]


def make_heat_map():
    m = folium.Map(location=[40.7, -74.05], zoom_start=11)
    subset = data[["latitude", "longitude"]][:].values.tolist()
    m.add_child(HeatMap(subset, radius=7.5))
    return m


def make_time_series(data_mat):
    time_accidents = np.sum(data_mat, axis=0)
    smooth_accidents = scipy.signal.savgol_filter(time_accidents, 61, 3)
    time = list(range(len(time_accidents)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time, y=time_accidents, mode="lines", name="Raw accidents"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=smooth_accidents, mode="lines", name="Smooth accidents"
        )
    )
    fig.show()


def make_mean_log_mean(data_mat):
    mean_accidents = np.sum(data_mat, axis=1) / len(data_mat)
    sns.displot(
        pd.DataFrame({"mean accidents": mean_accidents}),
        x="mean accidents",
        kind="kde",
    )
    sns.displot(
        pd.DataFrame(
            {
                "log mean accidents": np.log(
                    mean_accidents + 0.000000000001 / len(mean_accidents)
                )
            }
        ),
        x="log mean accidents",
        kind="kde",
    )


def plot_svi_loss(losses):
    elbo_df = pd.DataFrame(
        {"Iteration": list(range(len(losses))), "Loss": np.log(losses)}
    )
    fig = px.line(elbo_df, x="Iteration", y="Loss", title="Elbo")
    fig.show()


def plot_mean_variance_line(data_mat):
    """
    Plots variances across sites
    """
    selection = np.var(data_mat, axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(np.var(data_mat, axis=1), ax=ax[0])
    ax[0].set_xlabel("Variance")
    ax[0].set_title("Variance among sites")

    ax[1].set_ylabel("Variance")
    ax[1].set_xlabel("Mean")
    ax[1].set_title("Mean vs Variance per Site")

    x_values = np.linspace(0, 0.35, 1000)
    ax[1].plot(x_values, x_values, c="r", label="Line y = x")
    fig.show()
