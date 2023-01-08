import json
import pandas as pd
from paths import cars_path
from sklearn.metrics import mean_squared_error, mean_absolute_error


# avg_speeds = []
# with open("example.log", "r") as fp:
#     for line in fp:
#         line_dict = json.loads(line[10:])
#         if "avgSpeed" in line_dict:
#             avg_speeds.append(line_dict)

# df = pd.DataFrame(avg_speeds)
# df.head()
# df.to_csv("log_avg.csv", index=False)

cars = pd.read_csv(cars_path + "cars.csv")
estimation = pd.read_csv("log_avg.csv")
estimation = estimation[estimation.frameId > 249]
estimation['avgSpeed'] = pd.to_numeric(estimation['avgSpeed'], errors='coerce')

def avg_speed_for_time_ground_truth(timeStart, timeEnd):
    cars_to_avg = cars.loc[cars['start'].gt(timeStart) & cars['end'].le(timeEnd)]
    return cars_to_avg['speed'].mean()

def avg_speed_for_time_estimation(timeStart, timeEnd):
    timeStart *= 50
    timeEnd *= 50
    estimation_avg = estimation.loc[estimation['frameId'].gt(timeStart) & estimation['frameId'].le(timeEnd)]
    return estimation_avg['avgSpeed'].mean()


truth = []
predicted = []
# 54min * 60s = 3.240s
for start in range(250, 3240, 60):
    end = start + 60
    truth.append(avg_speed_for_time_ground_truth(start, end))
    predicted.append(avg_speed_for_time_estimation(start,end))

print("Truth vs. Predicted 50min divided in 1min slots: ")
print(f"mean_squared_error: {mean_squared_error(truth, predicted)}")
print(f"mean_absolute_error: {mean_absolute_error(truth, predicted)}")
