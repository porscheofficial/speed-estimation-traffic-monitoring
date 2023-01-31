from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import norm
from dataclasses import dataclass


@dataclass
class CameraPoint:
    frame: NDArray
    u: int
    v: int

    def coords(self):
        return self.u, self.v


@dataclass
class WorldPoint:
    frame: NDArray
    x: float
    y: float
    z: float

    def coords(self):
        return self.x, self.y, self.z


class GroundTruthEvent(NamedTuple):
    coords1: tuple[int, int, int]
    coords2: tuple[int, int, int]
    distance: float


class DepthModel:
    predict_function = lambda x: x

    def predict_depth(self, frame: NDArray) -> NDArray:
        # predict depth here
        depth_map = self.predict_function(frame)
        return depth_map


@dataclass
class GeometricModel:
    depth_model: DepthModel
    f: float = 1  # focal length
    s_u: int = 1  # translating pixels into m in u direction
    s_v: int = 1  # translating pixels into m in v direction
    c_u: int = 0  # this would usually be chosen at half the frame resolution width
    c_v: int = 0  # this would usually be chosen at half the frame resolution height
    scale_factor: float = 1

    def get_unscaled_world_point(self, cp: CameraPoint) -> WorldPoint:
        normalised_u = self.s_u * (cp.u - self.c_u)
        normalised_v = self.s_v * (cp.v - self.c_v)

        _, theta, phi = self._cartesian_to_sperhical(
            x=normalised_u, y=normalised_v, z=self.f
        )

        depth_map = self.depth_model.predict_depth(cp.frame)
        unscaled_depth = depth_map[cp.u, cp.v]

        return WorldPoint(
            frame=cp.frame,
            *self._spherical_to_cartesian(r=unscaled_depth, theta=theta, phi=phi)
        )

    def get_world_point(self, cp: CameraPoint) -> WorldPoint:
        unscaled_world_point = self.get_unscaled_world_point(cp)
        unscaled_world_point.x *= self.scale_factor
        unscaled_world_point.y *= self.scale_factor
        unscaled_world_point.z *= self.scale_factor

        return unscaled_world_point

    def get_camera_point(self, wp: WorldPoint) -> CameraPoint:
        r, theta, phi = self._cartesian_to_sperhical(*wp.coords())
        u, v, z_hat = self._spherical_to_cartesian(
            r=self.f / (np.sin(theta) * np.cos(phi)), theta=theta, phi=phi
        )

        assert z_hat == self.f

        return CameraPoint(frame=wp.frame, u=u, v=v)

    @staticmethod
    def _spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def _cartesian_to_sperhical(x, y, z):
        r = norm(x, y, z)
        theta = np.arccos(z / r)
        phi = np.sign(y) * np.arccos(x / norm(x, y))
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

        # once calibration is finished, we can start using the geometric_model to perform actual predictions for
        # velocities, however, even then we can still continue updating the scale factor
