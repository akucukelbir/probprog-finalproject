
"""
These are just some functions to make pretty graphs for the PPCs
"""

from statsmodels.nonparametric.kde import KDEUnivariate
import pyro
from pyro import plate
import pyro.distributions as dist
import pyro.contrib.autoguide as autoguide
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
import pyro.optim as optim
import numpy as np
import scipy.signal as signal
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import datetime
from sklearn.neighbors import KernelDensity
import folium
import seaborn as sns
from folium.plugins import FastMarkerCluster, HeatMap




def plot_max(samples, train_mat, num_max=10):
    """
    Parameters
    samples ndarray: shape [samples, sites, days]
    train_mat: ndarray shape [sites, days]
    num_max: number of max elements to plot
    """
    ax = plt.subplot(111)
    ax.set_title("Maximum number of accidents")
    
    max_vals = [np.max(elem) for elem in samples]
    sns.histplot(max_vals, ax = ax, stat='probability')
    ax.scatter(x=[np.sort(train_mat.flatten())[-num_max:]],
               y=[0]*num_max,
               s=100,c='g',alpha=1, label="Real max {}".format(num_max))
    ax.legend()
    return ax 



def plot_total_distributions(samples, train_mat, shape, subset):
    """
    Plots the  histplot of the total crashes per site on selected sites alongside real values. 

    Parameters: 
    samples (ndarray): samples[i][j][k] number of crashes in node j on day k
        on posterior sample i
    train_mat (ndarray): train_mat[i][j] crashes on site i on day j 
    shape (tuple): Shape of plots shape[0] numrows shape[1] is numcols
    subset (array): list containing subset of sites to look at 

    Returns:
    fig
    """
    to_plot = [np.sum(samples[:,i], axis = -1) for i in subset]
    real_total = np.sum(train_mat[subset],axis=-1)
    
    fig, axs = plt.subplots(nrows=shape[0],ncols=shape[1], figsize=(shape[1]*4,3*shape[0])) 
    fig.suptitle("Total crash distribution at different sites")
    for i in range(len(axs)): 
        for j in range(len(axs[0])):
            sns.histplot(x=to_plot[i+j],ax=axs[i][j], stat='probability',fill=False)
            axs[i][j].scatter(x=[real_total[i+j]],y=[0], alpha= 1, marker='o', label='Dataset total', c='r',s=100)
            axs[i][j].legend()
    fig.tight_layout()
    
    
def plot_time_trend(samples, train_mat, window=61, polynomial=3):
    """
    Plots the 95, 80, 50 confidence intervals for the svagol smoother.
    
    Params:
    window int: Window for svagol filter
    polynomial int: poly to use in savgol filter.
    samples (ndarray): samples[i][j][k] number of crashes in node j on day k on posterior
        sample i.
    train_mat (ndarray): train_mat[i][j] crashes on site i on day j 
    
    returns: 
        retuns plotted axis
    """

    time_series = np.sum(samples, axis=1)
    smooth_time_series = [signal.savgol_filter(series,window,polynomial) for series in time_series]
    smooth_time_series = np.array(smooth_time_series)
    quantiles_to_observe = [0.025,0.15,0.40,0.50,0.85,0.975]
    print(smooth_time_series.shape)
    quantiles = np.quantile(smooth_time_series, quantiles_to_observe, axis=0)
    colors = ['r','g','b']
    real = signal.savgol_filter(np.sum(train_mat, axis=0),window,polynomial)
    time = range(len(real))
    fig, ax = plt.subplots(figsize=(10,8),dpi=300)
    ax.plot(time, real, c= 'black', label='Original dataset')
    for i in range(int(len(quantiles_to_observe)/2)): 
        alpha = quantiles_to_observe[i]
        ax.fill_between(time, quantiles[i],quantiles[-i - 1],
                        alpha = 0.25, color=colors[i],
                        label='Posterior {:.2f} interval'.format(1-2*alpha))
   
    ax.set_title("Savgol smoothed accidents per day in Manhattan")
    ax.legend()
    return ax
    
