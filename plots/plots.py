import sys

sys.path.insert(1, "../")

from folium.plugins import FastMarkerCluster, HeatMap
import folium
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

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
    fig.update_layout(
        title="Accidents per day in Manhattan",
        xaxis_title="Days since start of 2014",
        yaxis_title="Number of accidents",)
    fig.show()


def make_mean_log_mean(data_mat):
    mean_accidents = np.sum(data_mat, axis=1) / len(data_mat)
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sns.kdeplot(mean_accidents, ax= ax[0])
    sns.kdeplot(np.log(mean_accidents + 0.000000001), ax= ax[1])
    ax[0].set_xlabel("mean accidents per day for each site")
    ax[1].set_xlabel("log of mean accidents per day for each site ")
    fig.suptitle("Means and log means accidents of each site per day")

def plot_svi_loss(losses):
    elbo_df = pd.DataFrame(
        {"Iteration": list(range(len(losses))), "log-loss": np.log(losses)}
    )
    fig = px.line(elbo_df, x="Iteration", y="log-loss", title="Elbo")
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
    variances = np.var(data_mat,axis=1)
    means = np.mean(data_mat,axis=1)
    ax[1].scatter(means, variances)
    ax[1].plot(x_values, x_values, c="r", label="Line y = x")


def plot_betas(beta_samples, pred_names):
    """
    Plots the graphs corresponding for the betas obtained from 
    """
    pred_included = [constants.PRED_DICT[pred_name] for pred_name in pred_names]
    sorted_values = sorted(zip(pred_included, pred_names), key = lambda x: x[0])

    fig, ax = plt.subplots(nrows=len(pred_names),ncols=1, figsize = (10, 3 * len(pred_names)), sharex=True)
    fig.suptitle('Posterior distributions for Predictors ')
                           
    
    for i in range(len(sorted_values)):
        sns.kdeplot(
            beta_samples[:,i+1],
            shade=True,
            ax=ax[i]
        )
        ax[i].set_title(sorted_values[i][1])
    plt.tight_layout()

    fig.show()
    
    
    
