import numpy as np
import pandas as pd

def process_aadt():
    aadt_complete = pd.read_csv("intersection/raw/aadt.csv", index_col=False)
    aadt_complete = aadt_complete.drop_duplicates().dropna(subset=['Count'])
    
    aadt_2019_map = pd.read_csv("intersection/raw/aadt_2019_map.csv", index_col=0)
    aadt_2019_map = aadt_2019_map.rename(columns={'RoadwayName': 'Road Name', 'BeginDescription': 'Beginning Description', 'EndDescription': 'Ending Description'})

    #We only care about the last 6 years
    years = aadt_complete.Year.unique()
    years_aadt = []

    for year in years:
        years_aadt.append(aadt_complete[:][aadt_complete.Year == year])
    years = years[:6]
    years_aadt = years_aadt[:6]

    #Process each individual year dataframes
    for i, year in enumerate(years):
        years_aadt[i] = years_aadt[i].rename(columns={'Count': 'Count_' + str(year)})
        columns_to_drop = ['Year']
        if i > 0:
            columns_to_drop = ['Ramp', 'Bridge', 'Railroad Crossing', 'Year', 'Station ID', 'Municipality', 'One Way', 'Functional Class', 'Length', 'County', 'Signing', 'State Route', 'County Road']
        years_aadt[i] = years_aadt[i].drop(columns_to_drop, axis=1)

    #Join the dataframes
    joined_aadt = years_aadt[0]
    for year_aadt in years_aadt[1:]:
        joined_aadt = pd.merge(year_aadt.drop_duplicates(), joined_aadt, how='outer', on=['Road Name', 'Beginning Description', 'Ending Description'])

    joined_aadt = pd.merge(joined_aadt, aadt_2019_map, how='inner', on=['Road Name', 'Beginning Description', 'Ending Description'])
    joined_aadt.to_csv("intersection/intermediate/joined_aadt.csv", index=False)

