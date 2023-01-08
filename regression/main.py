import os
from regression import get_pixel_length_of_car

base_path = "data/yolov5/runs/detect/exp3/crops/cars/"

for idx, filename in enumerate(sorted(os.listdir(base_path))):
    #background_removal(filename)
    #cnn(filename)
    car_length = get_pixel_length_of_car(base_path + filename)
