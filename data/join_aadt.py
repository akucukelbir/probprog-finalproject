#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


aadt_complete = pd.read_csv("aadt.csv", index_col=False)
print(aadt_complete.size)
aadt_complete = aadt_complete.drop_duplicates().dropna(subset=['Count'])
print(aadt_complete.size)
aadt_2019_map = pd.read_csv("aadt_2019_map.csv", index_col=0)
aadt_2019_map = aadt_2019_map.rename(columns={'RoadwayName': 'Road Name', 'BeginDescription': 'Beginning Description', 'EndDescription': 'Ending Description'})
print(aadt_2019_map.columns)
# In[18]:


years = aadt_complete.Year.unique()
years_aadt = []

for year in years:
    years_aadt.append(aadt_complete[:][aadt_complete.Year == year])
years = years[:6]
years_aadt = years_aadt[:6]


# In[14]:


aadt_complete.columns


# In[19]:


for i, year in enumerate(years):
    years_aadt[i] = years_aadt[i].rename(columns={'Count': 'Count_' + str(year)})
    columns_to_drop = ['Year']
    if i > 0:
        columns_to_drop = ['Ramp', 'Bridge',
       'Railroad Crossing', 'Year', 'Station ID', 'Municipality', 'One Way', 'Functional Class', 'Length', 'County', 'Signing', 'State Route', 'County Road']
    years_aadt[i] = years_aadt[i].drop(columns_to_drop, axis=1)


# In[20]:


joined_aadt = years_aadt[0]
for year_aadt in years_aadt[1:]:
    print(joined_aadt)
    print(year_aadt)
    joined_aadt = pd.merge(year_aadt.drop_duplicates(), joined_aadt, how='outer', on=['Road Name', 'Beginning Description', 'Ending Description'])
    print(joined_aadt)
    print("AS")

joined_aadt = pd.merge(joined_aadt, aadt_2019_map, how='inner', on=['Road Name', 'Beginning Description', 'Ending Description'])
print(joined_aadt)
joined_aadt.to_csv("joined_aadt.csv", index=False)
