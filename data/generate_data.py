#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[17]:


# These parameters were determined by looking at a map
LOCATION = [40.7128, -74.0060]
LOWER_LEFT = [LOCATION[0] - 0.3, LOCATION[1] - 0.3]
UPPER_RIGHT = [LOCATION[0] + 0.3, LOCATION[1] + 0.4]
FILEPATH = 'reduced_data.csv'

#NUMBER of ways to divide the grid by 
NUM_DIVISIONS = [10, 100, 500, 1000]

df = pd.read_csv(FILEPATH, index_col=0)


# In[18]:


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
    
    df_copy['x_box'] = np.floor((df_copy['longitude'] - xleft)/xstride)
    df_copy['y_box'] = np.floor((df_copy['latitude'] - ybot)/ystride)
    df_copy
    df_copy['repr_lat'] = df_copy['y_box']* ystride + ystride/2 + ybot
    df_copy['repr_long'] = df_copy['x_box']* xstride + xstride/2 + xleft
    
    return df_copy
    


# In[21]:


for n in NUM_DIVISIONS:
    aggregate_data(df, LOWER_LEFT, UPPER_RIGHT, n).to_csv("aggregated_data_{}_divisions.csv".format(n),index=False)
    print("Done generating data with {} divisions".format(n))


# In[ ]:




