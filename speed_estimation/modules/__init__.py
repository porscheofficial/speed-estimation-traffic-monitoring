"""Modules that are used in the speed estimation pipeline are defined in this module."""

from .depth_map.depth_map_utils import DepthModel
from .object_detection.yolov4.object_detection import (
    ObjectDetection as ObjectDetectionYoloV4,
)
from .shake_detection.shake_detection import ShakeDetection
