#!/usr/bin/env python
# coding: utf-8

# In[31]:


path = '../datasets/video.mp4'
#give_me_fps(path)


# In[30]:


import numpy as np
import pandas as pd
import cv2
import time
#import ffmpeg
import os

def give_me_fps(path_to_dataset):
    fps_a = read_fps_cv2(path_to_dataset)
    fps_b = read_fps_strange(path_to_dataset)
    fps = fps_a
    print("a:" + str(fps_a))
    print("b:" +str(fps_b))

    if fps > 80:
        fps = fps_b
        if fps > 80:
            fps = 25 #average of most videos
    
    print("final:" +str(fps))
    return int(fps)


# In[2]:


def read_fps_cv2(path_to_dataset):
    video = cv2.VideoCapture(path_to_dataset)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        #print (fps)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        #print (fps)

    video.release()
    return fps


# In[10]:


def read_fps_strange(path_to_dataset):
 
 
    # creating the videocapture object
    # and reading from the input file
    # Change it to 0 if reading from webcam

    cap = cv2.VideoCapture(path_to_dataset)

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # Reading the video file until finished
    fps_list = []
    counter = 0
    while(cap.isOpened()):
        counter +=1

        if counter ==20:
            break

        # Capture frame-by-frame

        ret, frame = cap.read()

        # if video finished or no Video Input
        if not ret:
            break

        # Our operations on the frame come here
        gray = frame

        # resizing the frame size according to our need
        # gray = cv2.resize(gray, (500, 300))

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)
        fps = fps #since twice as much normal idk

        fps_list.append(fps)
        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        #print(fps)


        # putting the FPS count on the frame
        #cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        # displaying the frame with fps
        #cv2.imshow('frame', gray)

        # press 'Q' if you want to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # Destroy the all windows now

    avg_fps = sum(fps_list)/len(fps_list)
    #print(avg_fps)
    return avg_fps

