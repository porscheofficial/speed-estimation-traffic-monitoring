import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.optimize import curve_fit
from scipy.spatial import distance
from sklearn import linear_model


def objective(x, a, b, c):
    return a * x**2 + b * x + c


# def objective(x, a, b):
#     return a*x + b


def get_pixel_length_of_car(img_array):
    background_image = Image.fromarray(img_array)
    interpolated_image = Image.blend(
        background_image,
        Image.new(background_image.mode, background_image.size, "black"),
        0.92,
    ).convert(mode="1")
    foreground_image = ImageOps.invert(interpolated_image)

    df = pd.DataFrame(
        (
            (x, -y)
            for x in range(foreground_image.width)
            for y in range(foreground_image.height)
            if not foreground_image.getpixel((x, y))
        ),
        columns=("x", "y"),
    )
    # Plot
    # df.plot.scatter(x=0, y=1, s=5, alpha=.5, c=["#111133"], figsize=(5,5)).set_aspect("equal")

    x = df["x"].values
    y = df["y"].values

    length = len(df["x"])
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    predictions = regr.predict(x)

    # plt.plot(x, predictions, color='green', linewidth=3)

    keep_smallest_y = df.loc[df.groupby("x").y.idxmin()]
    # plt.scatter(keep_smallest_y['x'].values, keep_smallest_y['y'].values,  color='green')

    test_x = keep_smallest_y["x"].values
    test_y = keep_smallest_y["y"].values
    # test_x = df['x'].values
    # test_y = df['y'].values
    popt, _ = curve_fit(objective, test_x, test_y)
    a, b, c = popt
    # plt.scatter(test_x, test_y, label="raw data")
    # b_values = y - a * x
    c_values = y - a * x**2 + b * x

    # plt.plot(test_x, objective(test_x, a, b, c), '--', color='red', label="lower bound", linewidth=4)
    # plt.plot(test_x, objective(test_x, a, np.max(b_values)), '--', color='orange', label="upper bound")
    # plt.legend()
    idx = np.argmin(objective(test_x, a, b, c))

    x_value_to_cut = df.iloc[df[["y"]].idxmin()]["x"].values[0]

    # we found the x value of the lowest point in the image (x_value_to_cut), so lets cut the image here
    # if the slope is positive, the camera sees the right side of the car and vise versa
    if len(test_x) < 100:
        return None

    if regr.coef_[0][0] > 0:
        df_cutted = df[df["x"] > x_value_to_cut]

        idx += 50
        min_point = (
            test_x[min(idx, len(test_x) - 1)],
            objective(test_x, a, b, c)[min(idx, len(test_x) - 1)],
        )
        max_point = (test_x[-1], objective(test_x, a, b, c)[-1])
        # plt.plot([test_x[idx],test_x[-1]], [objective(test_x, a, b, c)[idx],objective(test_x, a, b, c)[-1]], '--', color='yellow', label="lower bound", linewidth=4)
        # print("Estimated distance: ", distance.euclidean(min_point, max_point))

    else:
        df_cutted = df[df["x"] < x_value_to_cut]
        # empirical evidence has shown...
        idx -= 25
        min_point = (
            test_x[min(idx, len(test_x) - 1)],
            objective(test_x, a, b, c)[min(idx, len(test_x) - 1)],
        )
        max_point = (test_x[0], objective(test_x, a, b, c)[0])
        # plt.plot([test_x[idx],test_x[0]], [objective(test_x, a, b, c)[idx],objective(test_x, a, b, c)[0]], '--', color='yellow', label="lower bound", linewidth=4)
        # print("Estimated distance: ", distance.euclidean(min_point, max_point))

    return distance.euclidean(min_point, max_point)
    # if subtracted_df_cutted has no elements, the assumption of how the camera look onto the car was wrong
    # therefore we cut the image in the middle, but turn the logic around --> if the slope is positive the camera captures the car from the left side
    if len(df_cutted) == 0:
        if regr.coef_[0][0] > 0:
            df_cutted = df[df["x"] < foreground_image.width // 2]
        else:
            df_cutted = df[df["x"] > foreground_image.width // 2]

    x = df_cutted["x"].values
    y = df_cutted["y"].values

    length = len(df_cutted["x"])
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)

    regr.fit(x, y)
    predictions = regr.predict(x)

    regr_line_coordinates = list(zip(x.reshape(-1), predictions.reshape(-1)))

    # previously calculated manually: math.sqrt((abs(predictions[0]) - abs(predictions[-1]))**2 + (x[0][0] - x[-1][0])**2)
    car_length_in_pixels = distance.euclidean(regr_line_coordinates[0], regr_line_coordinates[-1])

    plt.scatter(x, y, color="black")
    plt.plot(x, predictions, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return car_length_in_pixels
