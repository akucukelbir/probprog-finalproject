import numpy as np
import pandas as pd
import math
import datetime

data_2013 = pd.read_csv("2013_15_weather.csv")
data_2016 = pd.read_csv("2016_18_weather.csv")
data_2019 = pd.read_csv("2019_weather.csv")

df_jfk_2013 = data_2013[data_2013['NAME'] == "JFK INTERNATIONAL AIRPORT, NY US"]
df_jfk_2016 = data_2016[data_2016['NAME'] == "JFK INTERNATIONAL AIRPORT, NY US"]
df_jfk_2019 = data_2019[data_2019['NAME'] == "JFK INTERNATIONAL AIRPORT, NY US"]
df_jfk = pd.concat([df_jfk_2013, df_jfk_2016, df_jfk_2019])
weather = df_jfk[['DATE', 'AWND', 'PRCP', 'SNWD', 'TAVG']]
weather = weather[pd.to_datetime(weather['DATE']) >= pd.to_datetime(datetime.date(2014, 1, 1))]
weather.to_csv('weather.csv')
