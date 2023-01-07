import pandas as pd
import tensorflow as tf
from matplotlib import pyplot
from numpy import expand_dims
from matplotlib import pyplot
import cv2
import os
import numpy as np
from contours import draw_contours, draw_circles, draw_circles_sklearn, get_pixel_length_of_carr


def background_removal(filename):
    # Read image
    img = cv2.imread("extracted_cars/" + filename)

    # threshold on white
    # Define lower and uppper limits
    lower = np.array([100, 100, 100])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    # save results
    cv2.imwrite('background_removal/thresh.png', thresh)
    cv2.imwrite('background_removal/morph.png', morph)
    cv2.imwrite('background_removal/mask.png', mask)
    cv2.imwrite('background_removal/result.png', result)


def cnn(filename):
    WIDTH = 1280
    HEIGHT = 720

    #Load the model
    model = tf.keras.applications.VGG16()

    # Summary of the model
    model.summary()

    # for i in range(len(model.layers)):
    #     layer = model.layers[i]
    #     if 'conv' not in layer.name:
    #         continue    
    #     print(i , layer.name , layer.output.shape)

    model = tf.keras.Model(inputs=model.inputs , outputs=model.layers[1].output)

    image = tf.keras.preprocessing.image.load_img("extracted_cars/" + filename , target_size=(224,224))

    # convert the image to an array
    image = tf.keras.preprocessing.image.img_to_array(image)
    # expand dimensions so that it represents a single 'sample'
    image = expand_dims(image, axis=0)

    image = tf.keras.applications.vgg16.preprocess_input(image)

    img_array = tf.keras.preprocessing.image.img_to_array(image.squeeze())
    # save the image with a new filename
    tf.keras.preprocessing.image.save_img('preprocessed.png', img_array)

    #calculating features_map
    features = model.predict(image)

    fig = pyplot.figure(figsize=(20,15))
    for i in range(1,features.shape[3]+1):
        pyplot.subplot(8,8,i)
        pyplot.imshow(features[0,:,:,i-1] , cmap='gray')
    
    pyplot.savefig('vgg_' + filename)
    pyplot.close('all')
    
    high_contrast_image = features[0,:,:,61]
    high_contrast_image = np.array(high_contrast_image)[..., None]
    cv2.imwrite('test_hc.png', high_contrast_image)

    image_hc = cv2.imread('test_hc.png')
    original_image_hc = image_hc.copy()

    hsv = cv2.cvtColor(image_hc, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 118])
    upper = np.array([179, 255, 202])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(original_image_hc,original_image_hc,mask=mask)

    cv2.imwrite('mask.png', mask)
    cv2.imwrite('mask_test.png', mask[91:223, :])
    cv2.imwrite('result.png', result)
    cv2.imwrite('original.png', original_image_hc)


    # mask_image = tf.keras.preprocessing.image.load_img("mask_test.png" , target_size=(92,224))

    # # convert the image to an array
    # mask_image = tf.keras.preprocessing.image.img_to_array(mask_image)
    # # expand dimensions so that it represents a single 'sample'
    # mask_image = expand_dims(mask_image, axis=0)

    # mask_image = tf.keras.applications.vgg16.preprocess_input(mask_image)

    # mask_features = model.predict(mask_image)

    # fig = pyplot.figure(figsize=(20,15))
    # for i in range(1,features.shape[3]+1):

    #     pyplot.subplot(8,8,i)
    #     pyplot.imshow(mask_features[0,:,:,i-1] , cmap='gray')
    
    # pyplot.savefig('vgg_mask_' + filename)
    # pyplot.close('all')

    # high_contrast_image_mask = features[0,:,:,61]
    # high_contrast_image_mask = tf.keras.preprocessing.image.img_to_array(high_contrast_image_mask, dtype='uint8')

    #draw_contours()
    #draw_circles(filename)
    #draw_circles_sklearn(filename)
    #car_length = get_pixel_length_of_carr()

base_path = "yolov5/runs/detect/exp10/crops/cars/"

for idx, filename in enumerate(sorted(os.listdir(base_path))):
    #background_removal(filename)
    #cnn(filename)
    car_length = get_pixel_length_of_carr(base_path + filename)