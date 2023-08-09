"""Configurations for the paths and directories.

All paths that are needed for the speed estimation are defined in this file.
The paths are all relative to `speed_estimation/`.
"""

# Paths for the object detection
PATH_TO_HAAR_FILE: str = "model_weights/myhaar.xml"
YOLOV4_WEIGHTS: str = "model_weights/yolov4.weights"
YOLOV4_CLASSES: str = "model_weights/classes.txt"
YOLOV4_CONFIG: str = "model_weights/yolov4.cfg"

# Path to the directory where the video is located.
SESSION_PATH: str = "../datasets"
VIDEO_NAME: str = "video.mp4"

SPEED_ESTIMATION_CONFIG_FILE: str = "config.ini"
