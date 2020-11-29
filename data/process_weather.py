import numpy as np
import pandas as pd
import math
import datetime

def process(location_name, columns, start_date):
    data_2013 = pd.read_csv("weather/raw/2013_15_weather.csv")
    data_2016 = pd.read_csv("weather/raw/2016_18_weather.csv")
    data_2019 = pd.read_csv("weather/raw/2019_weather.csv")

    df_jfk_2013 = data_2013[data_2013['NAME'] == location_name]
    df_jfk_2016 = data_2016[data_2016['NAME'] == location_name]
    df_jfk_2019 = data_2019[data_2019['NAME'] == location_name]
    df_jfk = pd.concat([df_jfk_2013, df_jfk_2016, df_jfk_2019])
    
    weather = df_jfk[columns]
    weather = weather[pd.to_datetime(weather['DATE']) >= pd.to_datetime(start_date)]
    weather.to_csv('weather/processed/data.csv')
