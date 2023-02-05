import cv2
from enum import Enum
from typing import NamedTuple
import numpy as np
from dataclasses import dataclass

def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 

class Direction(Enum):
    towards = 0
    away = 1

@dataclass
class Point:
    x: int
    y: int

    def coords(self):
        return np.array([self.x, self.y])

class Line(NamedTuple):
        start: Point
        end: Point

class TrackingBox:
    def __init__(self, center_x, center_y, x, y, w, h, frame, object_id=None) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame
        self.object_id = object_id

class Car:
    def __init__(self, tracked_boxes: list, frames_seen, frame_start, frame_end, direction:Direction = None) -> None:
        self.tracked_boxes = tracked_boxes
        self.frames_seen = frames_seen
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.direction = direction


def render_detected_frames_to_video(count, fps, out_video_name, path_to_frames):
    img_array = []
    for c in range(0, count):
        c += 1
        img = cv2.imread(path_to_frames % c)

        if img is None:
            continue

        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps,
                          size)  # fps have to get set automatically from orignal video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
