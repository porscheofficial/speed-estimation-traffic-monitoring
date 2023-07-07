from dataclasses import dataclass
from enum import Enum
from typing import List
from typing import NamedTuple

import cv2
import numpy as np


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


class Direction(Enum):
    TOWARDS = 0
    AWAY = 1


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
    def __init__(
        self,
        center_x: int,
        center_y: int,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_count: int,
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_count = frame_count


class Car:
    def __init__(
        self,
        tracked_boxes: List[TrackingBox],
        frames_seen: int,
        frame_start: int,
        frame_end: int,
        direction: Direction = None,
    ) -> None:
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

    out = cv2.VideoWriter(
        out_video_name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, size
    )  # fps have to get set automatically from orignal video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def get_intersection(line_a: Line, line_b: Line) -> Point:
    b = Point(*line_a.end.coords() - line_a.start.coords())
    d = Point(*line_b.end.coords() - line_b.start.coords())
    b_dot_d = b.x * d.y - b.y * d.x

    if b_dot_d == 0:
        # lines are parallel, no intersection
        return False

    c = Point(*line_b.start.coords() - line_a.start.coords())
    t = (c.x * d.y - c.y * d.x) / b_dot_d
    if t < 0 or t > 1:
        return False

    u = (c.x * b.y - c.y * b.x) / b_dot_d
    if u < 0 or u > 1:
        return False

    return Point(*line_a.start.coords() + t * b.coords())


def calculate_car_direction(car: Car) -> Direction:
    first_box = car.tracked_boxes[0]
    last_box = car.tracked_boxes[-1]

    if (first_box.y - last_box.y) < 0:
        return Direction.TOWARDS

    return Direction.AWAY
