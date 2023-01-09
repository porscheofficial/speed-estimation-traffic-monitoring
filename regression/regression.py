import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import pandas as pd
from sklearn import linear_model
from scipy.spatial import distance
import math

def get_pixel_length_of_car(path_to_image):
    # a = Image.open("mask.png")
    background_image = Image.open(path_to_image)
    interpolated_image = Image.blend(background_image, Image.new(background_image.mode, background_image.size, "black"), .92).convert(mode="1")
    foreground_image = ImageOps.invert(interpolated_image)

    df = pd.DataFrame(((x,-y) for x in range(foreground_image.width) for y in range(foreground_image.height) if not foreground_image.getpixel((x,y))), columns=("x","y"))
    # Plot
    df.plot.scatter(x=0, y=1, s=5, alpha=.5, c=["#111133"], figsize=(5,5)).set_aspect("equal")
        
    x = df['x'].values
    y = df['y'].values

    length = len(df['x'])
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    x_value_to_cut = df.iloc[df[['y']].idxmin()]['x'].values[0]

    # we found the x value of the lowest point in the image (x_value_to_cut), so lets cut the image here
    # if the slope is positive, the camera sees the right side of the car and vise versa
    if regr.coef_[0][0] > 0:
        df_cutted = df[df["x"] > x_value_to_cut]
    else:
        df_cutted = df[df["x"] < x_value_to_cut]
    
    # if subtracted_df_cutted has no elements, the assumption of how the camera look onto the car was wrong
    # therefore we cut the image in the middle, but turn the logic around --> if the slope is positive the camera captures the car from the left side
    if len(df_cutted) == 0:
        if regr.coef_[0][0] > 0:
            df_cutted = df[df["x"] < foreground_image.width // 2]
        else:
            df_cutted = df[df["x"] > foreground_image.width // 2]

    x = df_cutted['x'].values
    y = df_cutted['y'].values

    length = len(df_cutted['x'])
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)

    regr.fit(x, y)
    predictions = regr.predict(x)

    regr_line_coordinates = list(zip(x.reshape(-1), predictions.reshape(-1)))

    # previously calculated manually: math.sqrt((abs(predictions[0]) - abs(predictions[-1]))**2 + (x[0][0] - x[-1][0])**2)
    car_length_in_pixels = distance.euclidean(regr_line_coordinates[0], regr_line_coordinates[-1])

    plt.scatter(x, y,  color='black')
    plt.plot(x, predictions, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return car_length_in_pixels