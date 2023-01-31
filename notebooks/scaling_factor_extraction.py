import numpy as np
from numpy.typing import NDArray
from collections import namedtuple
from typing import NamedTuple
from scipy.stats import norm


CameraPoint = namedtuple("CameraPoint", "frame u v")
WorldPoint = namedtuple("WorldPoint", "frame x y z")


class GroundTruthEvent(NamedTuple):
    coords1: (int, int, int)
    coords2: (int, int, int)
    distance: float


class DepthModel:
    predict_function = lambda x: x

    def predict_depth(self, frame: NDArray) -> NDArray:
        # predict depth here
        depth_map = self.predict_function(frame)
        return depth_map


class GeometricModel:
    f: float  # focal length
    s_u: int  # translating pixels into m in u direction
    s_v: int  # translating pixels into m in v direction
    c_u: int  # this would usually be chosen at half the frame resolution width
    c_v: int  # this would usually be chosen at half the frame resolution height
    scale_factor: float
    depth_model: DepthModel

    def __init__(self, depth_model, f=1, s_u=1, s_v=1, c_u=0, c_v=0, scale_factor=1):
        self.depth_model = depth_model
        self.f = f
        self.s_u = s_u
        self.s_v = s_v
        self.c_u = c_u
        self.c_v = c_v
        self.scale_factor = scale_factor

    def get_unscaled_world_point(self, cp: CameraPoint) -> WorldPoint:
        normalised_u = self.s_u * (cp.u - self.c_u)
        normalised_v = self.s_v * (cp.v - self.c_v)
        distance_from_screen_center = norm(normalised_v, normalised_u)
        distance_from_pinhole = norm(distance_from_screen_center, self.f)

        theta = np.arccos(self.f / distance_from_pinhole)
        phi = (
            np.sign(normalised_v)
            * np.arccos
            * (normalised_u / distance_from_screen_center)
        )

        depth_map = self.depth_model.predict_depth(cp.frame)
        unscaled_depth = depth_map[cp.u, cp.v]

        return WorldPoint(
            frame=cp.frame,
            x=unscaled_depth * np.sin(theta) * np.cos(phi),
            y=unscaled_depth * np.sin(theta) * np.sin(phi),
            z=unscaled_depth * np.cos(theta),
        )

    def get_world_point(self, cp: CameraPoint) -> WorldPoint:
        unscaled_world_point = self.get_unscaled_world_point(cp)
        unscaled_world_point.x *= self.scale_factor
        unscaled_world_point.y *= self.scale_factor
        unscaled_world_point.z *= self.scale_factor

        return unscaled_world_point

    @staticmethod
    def calculate_distance_between_world_points(
        wp1: WorldPoint, wp2: WorldPoint
    ) -> float:
        to_coords = lambda wp: np.array([wp.x, wp.y, wp.z])
        return norm(to_coords(wp1), to_coords(wp2))

    def get_unscaled_distance_from_camera_points(
        self, cp1: CameraPoint, cp2: CameraPoint
    ):
        unscaled_wp1 = self.get_unscaled_world_point(cp1)
        unscaled_wp2 = self.get_unscaled_world_point(cp2)
        return self.calculate_distance_between_world_points(unscaled_wp1, unscaled_wp2)

    def get_distance_from_camera_points(self, cp1: CameraPoint, cp2: CameraPoint):
        return self.scale_factor * self.get_unscaled_distance_from_camera_points(
            cp1, cp2
        )


def offline_scaling_factor_estimation_from_least_squares(
    frames: list[NDArray],
    geometric_model: GeometricModel,
    ground_truths: list[GroundTruthEvent],
) -> float:
    unscaled_predictions = []
    labels = []

    for coords1, coords2, distance in ground_truths:
        f1, u1, v1 = coords1
        f2, u2, v2 = coords2
        cp1 = CameraPoint(frame=frames[f1], u=u1, v=v1)
        cp2 = CameraPoint(frame=frames[f2], u=u2, v=v2)
        unscaled_predictions.append(
            geometric_model.get_unscaled_distance_from_camera_points(cp1, cp2)
        )
        labels.append(distance)

    # return optimal scaling factor under least sum of squares estimator
    return np.dot(unscaled_predictions, labels) / np.dot(
        unscaled_predictions, unscaled_predictions
    )


def online_scaling_factor_estimation_from_least_squares(stream_of_events):

    counter = 0

    depth_model = DepthModel()
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
            prediction**2
        ) / counter
        mean_prediction_dot_distance = (
            1 - 1 / counter
        ) * mean_prediction_dot_distance + (prediction * true_distance) / counter

        geometric_model.scale_factor = (
            mean_prediction_dot_distance / mean_predictions_two_norm
        )

        # once calibration is finished, we can start using the geometric_model to perform actual predictions for velocities
        # however, even then we can still continue updating the scale factor
