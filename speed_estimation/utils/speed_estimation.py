"""
The utility functions and classes for the speed estimation pipeline.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List
from typing import NamedTuple
from numpy.typing import NDArray
from typing import Optional

import cv2
import numpy as np



class Direction(Enum):
    """This enum holds the driving directions that can be detected.

    TOWARDS: The vehicle is driving towards the camera recorded/recording the video/stream
    AWAY: The vehicle is driving away from the camera recorded/recording the video/stream
    """

    TOWARDS = 0
    AWAY = 1
    UNDEFINED = 2


@dataclass
class Point:
    """This class is used to create a point in an 2D space."""

    x_coord: int
    y_coord: int

    def coords(self) -> NDArray:
        """Get the x and y coordinate of the point.

        @return:
            The x and y coordinates of the point are returned.
        """

        return np.array([self.x_coord, self.y_coord])


class Line(NamedTuple):
    """This class defines a line with a start point (see Point) and an end point (see Point)"""

    start: Point
    end: Point


class TrackingBox:
    """
    This class stores all the relevant information to detect a car in a frame.
    A TrackingBox is always a rectangle.
    """

    def __init__(
            self,
            center_x: int,
            center_y: int,
            x_coord: int,
            y_coord: int,
            width: int,
            height: int,
            frame_count: int,
    ) -> None:
        """Init method for a TrackingBox

        @param center_x:
            The x coordinate of the center point of the TrackingBox.
        @param center_y:
            The y coordinate of the center point of the TrackingBox.
        @param x_coord:
            The x coordinate of the upper left corner of the TrackingBox.
        @param y_coord:
            The y coordinate of the upper left corner of the TrackingBox.
        @param width:
            The width of the TrackingBox (x-direction)
        @param height:
            The height of the TrackingBox (y-direction)
        @param frame_count:
            The count of the frame in which the TrackingBox was detected.
        """
        self.center_x = center_x
        self.center_y = center_y
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.width = width
        self.height = height
        self.frame_count = frame_count


class Car:
    """This class represents a car with all the relevant information for a speed estimation."""

    def __init__(
            self,
            tracked_boxes: List[TrackingBox],
            frames_seen: int,
            frame_start: int,
            frame_end: int,
            direction: Direction = Direction.UNDEFINED,
    ) -> None:
        """The init method for a car.

        @param tracked_boxes:
            The tracking boxes that belong to the car.
        @param frames_seen:
            The number of frames the car has been seen so far.
        @param frame_start:
            The count of the frame the car was detected the first time.
        @param frame_end:
            The count of the frame the car was detected the last time.
        @param direction:
            The direction the car is driving.
        """
        self.tracked_boxes = tracked_boxes
        self.frames_seen = frames_seen
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.direction = direction


def get_intersection(line_a: Line, line_b: Line) -> Optional[Point]:
    """Find the intersection of two lines

    @param line_a:
        First line.
    @param line_b:
        Second line.
    @return:
        Returns the intersection coordinates of line_a and line_b wrapped in a Point object.
    """
    b = Point(*line_a.end.coords() - line_a.start.coords())
    d = Point(*line_b.end.coords() - line_b.start.coords())
    b_dot_d = b.x_coord * d.y_coord - b.y_coord * d.x_coord

    if b_dot_d == 0:
        # lines are parallel, no intersection
        return None

    c = Point(*line_b.start.coords() - line_a.start.coords())
    t = (c.x_coord * d.y_coord - c.y_coord * d.x_coord) / b_dot_d
    if t < 0 or t > 1:
        return None

    u = (c.x_coord * b.y_coord - c.y_coord * b.x_coord) / b_dot_d
    if u < 0 or u > 1:
        return None

    return Point(*line_a.start.coords() + t * b.coords())


def calculate_car_direction(car: Car) -> Direction:
    """Get the Direction the car isd driving.

    This function calculates in which the car is driving (towards or away from the camera).

    @param car:
        The car whose direction is to be determined.

    @return:
        The direction the car is driving.
    """
    first_box = car.tracked_boxes[0]
    last_box = car.tracked_boxes[-1]

    if (first_box.y_coord - last_box.y_coord) < 0:
        return Direction.TOWARDS

    return Direction.AWAY
