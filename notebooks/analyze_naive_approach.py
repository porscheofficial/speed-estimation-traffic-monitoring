from speed_estimation.modules.evaluation.evaluate import load_log
import json
import os
import re
import uuid
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import numpy as np
import math


log_path = '/home/mschneider/porsche_digital_hpi/logs/20230122-175423_run_8ceea506dc.log'
log_dict = []
speed_limit = 80


with open(log_path, "r") as fp:
    for idx, line in enumerate(fp):
        if idx == 0:
            result = re.search('Video: (.*), Max', line)
            cars_path = result.group(1)
            result = re.search('Max Depth: (.*)', line)
            max_depth = result.group(1)
            print(f"Found cars path from log: {cars_path}")
        if not line.startswith('INFO:root:{'):
            continue
        line_dict = json.loads(line[10:])
        if "car_id" in line_dict:
            log_dict.append(line_dict)
    
    df = pd.DataFrame(log_dict)

    df_grouped = df.groupby('car_id').agg(direction_indicator=('direction_indicator', 'sum'), frame_count=('frame_count', 'count'))

    df_grouped = df_grouped[df_grouped["frame_count"] > 50]
    df_grouped = df_grouped[df_grouped["direction_indicator"] < 0]

    avg_frame_count = df_grouped["frame_count"].mean()

    speeds = []

    for index, row in df_grouped.iterrows():
        speed_for_car = (avg_frame_count/row["frame_count"]) * speed_limit
        speeds.append(speed_for_car)
        print(str(index) + ": " + str(speed_for_car))
    
    df_grouped["speed"] = speeds
    print(df_grouped)