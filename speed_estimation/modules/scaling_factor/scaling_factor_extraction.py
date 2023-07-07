from dataclasses import dataclass
from typing import Dict, List
from typing import NamedTuple

import numpy as np
from modules.depth_map.depth_map_utils import DepthModel
from numpy.typing import NDArray
from scipy.linalg import norm
from scipy.spatial import distance
from utils.speed_estimation import Line, Point, TrackingBox, get_intersection


@dataclass
class CameraPoint:
    frame: int
    u: int
    v: int

    def coords(self) -> NDArray:
        return np.array([self.u, self.v])


@dataclass
class WorldPoint:
    frame: int
    x: float
    y: float
    z: float

    def coords(self) -> NDArray:
        return np.array([self.x, self.y, self.z])


class GroundTruthEvent(NamedTuple):
    coords1: tuple
    coords2: tuple
    distance: float


@dataclass
class GeometricModel:

    def __init__(self, depth_model) -> None:

        self.depth_model = depth_model
        self.f: float = 105.  # focal length
        self.s_u: int = 1  # translating pixels into m in u direction
        self.s_v: int = 1  # translating pixels into m in v direction
        self.c_u: int = 1  # this would usually be chosen at half the frame resolution width
        self.c_v: int = 1  # this would usually be chosen at half the frame resolution height
        self.scale_factor: float = 1

    def set_normalization_axes(self, c_u, c_v):
        self.c_u = c_u
        self.c_v = c_v

    def get_unscaled_world_point(self, cp: CameraPoint) -> WorldPoint:
        normalised_u = self.s_u * (cp.u - self.c_u)
        normalised_v = self.s_v * (cp.v - self.c_v)

        # we relabel the axis here to deal with different reference conventions
        _, theta, phi = self._cartesian_to_spherical(
            x=self.f, y=normalised_u, z=normalised_v
        )

        depth_map = self.depth_model.predict_depth(cp.frame)
        unscaled_depth = depth_map[cp.v, cp.u]

        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi

        x, y, z = self._spherical_to_cartesian(r=unscaled_depth, theta=theta, phi=phi)

        # relabeling again for the world reference system
        return WorldPoint(frame=cp.frame, x=y, y=z, z=x)

    def get_world_point(self, cp: CameraPoint) -> WorldPoint:
        unscaled_world_point = self.get_unscaled_world_point(cp)
        unscaled_world_point.x *= self.scale_factor
        unscaled_world_point.y *= self.scale_factor
        unscaled_world_point.z *= self.scale_factor

        return unscaled_world_point

    def get_camera_point(self, wp: WorldPoint) -> CameraPoint:

        x, y, z = wp.coords()
        # Note that we here relabel the coordinates to keep the two coordinate systems aligned!
        r, theta, phi = self._cartesian_to_spherical(x=z, y=x, z=y)
        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi
        z_inner, x_inner, y_inner = self._spherical_to_cartesian(
            r=np.abs(self.f / (np.sin(theta) * np.cos(phi))), theta=theta, phi=phi
        )

        assert np.isclose(z_inner, self.f)

        return CameraPoint(frame=wp.frame, u=x_inner, v=y_inner)

    @staticmethod
    def _spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def _cartesian_to_spherical(x, y, z):
        r = norm([x, y, z])
        if r == 0:
            return 0, 0, 0
        theta = np.arccos(z / r)
        if norm([x, y]) == 0:
            return z, 0, 0
        phi = np.sign(y) * np.arccos(x / norm([x, y]))
        return r, theta, phi

    @staticmethod
    def calculate_distance_between_world_points(
            wp1: WorldPoint, wp2: WorldPoint
    ) -> float:
        return norm(wp1.coords() - wp2.coords())

    def get_unscaled_distance_from_camera_points(
            self, cp1: CameraPoint, cp2: CameraPoint
    ):
        unscaled_wp1 = self.get_unscaled_world_point(cp1)
        unscaled_wp2 = self.get_unscaled_world_point(cp2)
        return self.calculate_distance_between_world_points(unscaled_wp1, unscaled_wp2)

    def get_distance_from_camera_points(self, cp1: CameraPoint, cp2: CameraPoint) -> float:
        return self.scale_factor * self.get_unscaled_distance_from_camera_points(
            cp1, cp2
        )


def offline_scaling_factor_estimation_from_least_squares(
        geometric_model: GeometricModel,
        ground_truths: list,
) -> float:
    unscaled_predictions = []
    labels = []

    for coords1, coords2, distance in ground_truths:
        f1, u1, v1 = coords1
        f2, u2, v2 = coords2
        cp1 = CameraPoint(frame=f1, u=u1, v=v1)
        cp2 = CameraPoint(frame=f2, u=u2, v=v2)
        unscaled_predictions.append(
            geometric_model.get_unscaled_distance_from_camera_points(cp1, cp2)
        )
        labels.append(distance)

    # return optimal scaling factor under least sum of squares estimator
    return np.dot(unscaled_predictions, labels) / np.dot(
        unscaled_predictions, unscaled_predictions
    )


def online_scaling_factor_estimation_from_least_squares(stream_of_events):
    ###################
    # TODO: integrate
    ###################
    counter = 0

    depth_model = DepthModel(data_dir="")
    geometric_model = GeometricModel(depth_model=depth_model)

    mean_predictions_two_norm = 0
    mean_prediction_dot_distance = 0

    while stream_of_events.has_next():
        counter += 1

        # calibration phase uses a stream of ground truth events
        frame, coords1, coords2, true_distance = stream_of_events.next()

        # this would e.g. be the pixel coordinates for the corners of a bounding box
        _, u1, v1 = coords1
        _, u2, v2 = coords2

        # calculate the unscaled predicted distance
        cp1 = CameraPoint(frame=frame, u=u1, v=v1)
        cp2 = CameraPoint(frame=frame, u=u2, v=v2)
        prediction = geometric_model.get_unscaled_distance_from_camera_points(cp1, cp2)

        mean_predictions_two_norm = (1 - 1 / counter) * mean_predictions_two_norm + (
                prediction ** 2
        ) / counter
        mean_prediction_dot_distance = (
                                               1 - 1 / counter
                                       ) * mean_prediction_dot_distance + (prediction * true_distance) / counter

        geometric_model.scale_factor = (
                mean_prediction_dot_distance / mean_predictions_two_norm
        )

        # once calibration is finished, we can start using the geometric_model to perform actual predictions for
        # velocities, however, even then we can still continue updating the scale factor


def get_ground_truth_events(tracking_boxes: Dict[int, List[TrackingBox]]):
    # extract medium pixel distance traveled by object
    box_distances = []
    for object_id in tracking_boxes:
        start_box = tracking_boxes[object_id][0]
        end_box = tracking_boxes[object_id][-1]
        tracking_box_distance = distance.euclidean([start_box.center_x, start_box.center_y],
                                                   [end_box.center_x, end_box.center_y])
        box_distances.append(tracking_box_distance)

    median_distance = np.percentile(np.array(box_distances), 50)

    # extract ground truth value for each tracking box
    ground_truth_events = []
    for object_id in tracking_boxes:
        center_points = np.array([(box.center_x, box.center_y) for box in tracking_boxes[object_id]])
        start_box = center_points[0]
        end_box = center_points[-1]
        tracking_box_distance = distance.euclidean(start_box, end_box)
        if len(center_points) < 2 or len(center_points) > 750 or tracking_box_distance < median_distance:
            continue
        center_points_line = Line(Point(*center_points[0]), Point(*center_points[-1]))

        # extract ground truth value for each tracking box
        for box in tracking_boxes[object_id]:

            # check each of the for lines, spanned by the bounding box rectangle
            upper_line = Line(Point(box.x, box.y), Point(box.x + box.w, box.y))
            right_line = Line(Point(box.x + box.w, box.y), Point(box.x + box.w, box.y + box.h))
            lower_line = Line(Point(box.x, box.y + box.h), Point(box.x + box.w, box.y + box.h))
            left_line = Line(Point(box.x, box.y), Point(box.x, box.y + box.h))

            intersections = []
            for bounding_box_line in [upper_line, right_line, lower_line, left_line]:
                intersection = get_intersection(center_points_line, bounding_box_line)
                if intersection:
                    intersections.append(intersection)

            if len(intersections) == 2:
                # append ground truth only if line fully cuts bounding box
                intersect1, intersect2 = intersections
                ground_truth_events.append(
                    GroundTruthEvent(
                        (box.frame_count, int(intersect1.x), int(intersect1.y)),
                        (box.frame_count, int(intersect2.x), int(intersect2.y)),
                        6
                    )
                )

    return ground_truth_events
