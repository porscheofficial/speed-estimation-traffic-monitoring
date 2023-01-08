import matplotlib.pyplot as plt

from PIL import Image
import pandas as pd
from sklearn import linear_model
import math

def get_pixel_length_of_car(path_to_image):
    # a = Image.open("mask.png")
    a = Image.open(path_to_image)
    b = Image.blend(a, Image.new(a.mode, a.size, "black"), .92).convert(mode="1")
    df = pd.DataFrame(((x,-y) for x in range(b.width) for y in range(b.height) if not b.getpixel((x,y))), columns=("x","y"))
    # Plot
    df.plot.scatter(x=0, y=1, s=5, alpha=.5, c=["#111133"], figsize=(16,16)).set_aspect("equal")
    
    full_list = []
    for x in range(0, b.width):
        for y in range(0, b.height):
            full_list.append((x, -y))
    
    full_df = pd.DataFrame(full_list, columns=['x', 'y'])

    subtracted_df = pd.concat([full_df, df]).drop_duplicates(keep=False)
    subtracted_df = subtracted_df.reset_index()
    
    x = subtracted_df['x'].values
    y = subtracted_df['y'].values

    length = len(subtracted_df['x'])
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    x_value_to_cut = subtracted_df.iloc[subtracted_df[['y']].idxmin()]['x']

    # we found the x value of the lowest point in the image (x_value_to_cut), so lets cut the image here
    # if the slope is positive, the camera sees the right side of the car and vise versa
    if regr.coef_[0][0] > 0:
        subtracted_df_cutted = subtracted_df[subtracted_df["x"] > x_value_to_cut.values[0]]
    else:
        subtracted_df_cutted = subtracted_df[subtracted_df["x"] < x_value_to_cut.values[0]]
    
    # if subtracted_df_cutted has no elements, the assumption of how the camera look onto the car was wrong
    # therefore we cut the image in the middle, but turn the logic around --> if the slope is positive the camera captures the car from the left side
    if len(subtracted_df_cutted) == 0:
        if regr.coef_[0][0] > 0:
            subtracted_df_cutted = subtracted_df[subtracted_df["x"] < 112]
        else:
            subtracted_df_cutted = subtracted_df[subtracted_df["x"] > 112]

    x = subtracted_df_cutted['x'].values
    y = subtracted_df_cutted['y'].values

    length = len(subtracted_df_cutted['x'])
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)

    regr.fit(x, y)

    predictions = regr.predict(x)

    car_length = math.sqrt((abs(predictions[0]) - abs(predictions[len(predictions)-1]))**2 + (x[0][0] - x[len(x)-1][0])**2)

    plt.scatter(x, y,  color='black')
    plt.plot(x, predictions, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return car_length