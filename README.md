# HPI for Porsche

## Structure
The config file of the application is called *config.ini* and stored inside the *object_detection_yolo* folder. There you can enable the custom object detection (yolov5) model. The default is the yolov4 model.
<br />
<br />
The different modules of this project can be found inside the folder *object_detection_yolo/modules*
You will find there:
- the object detection models
- the depth_map module to estimate the depth in the particular video
- the regression module, that is used to find the maximal depth of the video
- the evaluation module which compares our estimates with the ground truth


## Requirements
Before you start the code, please install the requirements.txt. Therefore, you can use this command:
`pip install -r docker/requirements.txt`
For being able to stream video data from the internet you need to install ffmpeg on your machine.

The path to the video that should be analysed can be set *object_detection_yolo/paths.py*

## How 2 Run
The easiest way to run the code right now is to use the debugger of visual studio code (at least this does work all on all our machines).
The main file to execute is `object_detection_yolo/object_tracking.py`
<br />