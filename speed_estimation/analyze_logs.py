import json
import re
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math
import configparser

def analyzer(log_file):
    config = configparser.ConfigParser()
    config.read('speed_estimation/config.ini')


    # add a new section and some values
    try:
        config.add_section('analyzer')
    except:
        print("")

    log_path = log_file
    log_dict = []
    speed_limit = int(config.get('analyzer', 'speed_limit'))
    #speed_limit = 80

    print(log_path)


    with open(log_path, "r") as fp:
        for idx, line in enumerate(fp):
            if not line.startswith('INFO:root:{'):
                continue
            line_dict = json.loads(line[10:])
            if "car_id" in line_dict:
                log_dict.append(line_dict)
        
        df = pd.DataFrame(log_dict)

        df_grouped = df.groupby('car_id').agg(direction_indicator=('direction_indicator', 'sum'), frame_count=('frame_count', 'count'))
        
        #outlier filtering
        df_grouped = df_grouped[df_grouped["frame_count"] > 20]
        df_grouped = df_grouped[df_grouped["direction_indicator"] != 0] #both directions

        avg_frame_count = df_grouped["frame_count"].mean()

        config.set('analyzer', 'avg_frame_count', str(avg_frame_count))

        speeds = []

        for index, row in df_grouped.iterrows():
            speed_for_car = (avg_frame_count/row["frame_count"]) * speed_limit
            speeds.append(speed_for_car)
            # print(str(index) + ": " + str(speed_for_car))
        
        df_grouped["speed"] = speeds
        print(df_grouped)

        with open('speed_estimation/config.ini', 'w') as configfile:
            config.write(configfile)

        return avg_frame_count, speed_limit
