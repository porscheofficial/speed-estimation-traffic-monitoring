from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from modules.depth_map.depth_map_utils import DepthModel
from numpy.typing import NDArray
from scipy.linalg import norm
from scipy.spatial import distance
from utils.speed_estimation import Line, Point, TrackingBox, get_intersection


@dataclass
class CameraPoint:
    """A Camera Point in the frame.

    A camera point is always two-dimensional, in a given frame and uses pixels as unit.

    @param frame
        The count of the frame the point belongs to.
    @param x_coord
        The x coordinate of the point in pixels.
    @param y_coord
        The y coordinate of the point in pixels.
    """

    frame: int
    x_coord: int
    y_coord: int

    def coords(self) -> NDArray:
        """Get coordinates of the CameraPoint.

        @return:
            Returns the x and y coordinate of the point.
        """
        return np.array([self.x_coord, self.y_coord])


@dataclass
class WorldPoint:
    """A three-dimensional world point.

    A world point is always three-dimensional and the projection of a CameraPoint into the real
    world.

    @param frame
        The count of the frame the point belongs to.
    @param x_coord
        The x coordinate of the point in pixels.
    @param y_coord
        The y coordinate of the point in pixels.
    @param z_coord
        The z coordinate of the point in pixels.
    """

    frame: int
    x_coord: float
    y_coord: float
    z_coord: float

    def coords(self) -> NDArray:
        """Get coordinates of WorldPoint.

        @return:
            Returns the x, y and z coordinate of the point.
        """
        return np.array([self.x_coord, self.y_coord, self.z_coord])


class GroundTruthEvent(NamedTuple):
    """A car tracked during the calibration phase.

    GroundTruth events are created from cars that have been tracked during the calibration phase.
    Cars and their tracking boxes are used as reference and calibration ground truth.

    @param coords1
        The first tuple holding the frame count, x coordinate and y coordinate
    @param coords2
        The second tuple holding the frame count, x coordinate and y coordinate
    @param distance
        The distance that lies between coord1 and coord2. Per default hard coded to 6m as this is
        a reasonable value for the ground truth length of a car.
    """

    coords1: Tuple
    coords2: Tuple
    distance: float


@dataclass
class GeometricModel:
    """Geometric model that is used to retrieve the distance between two points.

    This model hold the most important information to get the distance between two CameraPoints in
    meters.

    @param depth_model
        The depth model that has to be applied to predict the depth of the whole frame.
    @param focal_length
        Focal length of the camera recording the video/stream.
    @param scaling_x
        Scaling factor in x direction to translate pixels into meters.
    @param scaling _y
        Scaling factor in y direction to translate pixels into meters.
    @param center_x
        The center point in x direction.
    @param center_y
        The center point in y direction.
    """

    def __init__(self, depth_model) -> None:
        """Create an instance of GeometricModel.

        @param depth_model:
            The depth model that has to be applied to predict the depth of the whole frame.
        """
        self.depth_model = depth_model
        self.focal_length: float = 105.0
        self.scaling_x: int = 1
        self.scaling_y: int = 1
        self.center_x: int = 1
        self.center_y: int = 1
        self.scale_factor: float = 1

    def set_normalization_axes(self, center_x: int, center_y: int) -> None:
        """Set the normalization axis.

        @param center_x:
            The x coordinate of the center point.
        @param center_y:
            The y coordinate of the center point.
        """
        self.center_x = center_x
        self.center_y = center_y

    def get_unscaled_distance_from_camera_points(
        self, cp1: CameraPoint, cp2: CameraPoint
    ) -> float:
        """Get the unscaled distance between two two-dimensional CameraPoints.

        @param cp1:
            First CameraPoint.
        @param cp2:
            Second CameraPoint.
        @return:
            Returns the unscaled distance between those CameraPoints.
        """
        unscaled_wp1 = self.__get_unscaled_world_point(cp1)
        unscaled_wp2 = self.__get_unscaled_world_point(cp2)

        return self.__calculate_distance_between_world_points(
            unscaled_wp1, unscaled_wp2
        )

    def get_distance_from_camera_points(
        self, cp1: CameraPoint, cp2: CameraPoint
    ) -> float:
        """Get the scaled distance between two two-dimensional CameraPoints in meters.

        @param cp1:
            First CameraPoint.
        @param cp2:
            Second CameraPoint.
        @return:
            Returns the scaled distance between those CameraPoints in meters.
        """
        return self.scale_factor * self.__get_unscaled_distance_from_camera_points(
            cp1, cp2
        )

    def __get_unscaled_world_point(self, cp: CameraPoint) -> WorldPoint:
        normalised_u = self.scaling_x * (cp.x_coord - self.center_x)
        normalised_v = self.scaling_y * (cp.y_coord - self.center_y)

        # we relabel the axis here to deal with different reference conventions
        _, theta, phi = self.__cartesian_to_spherical(
            x=self.focal_length, y=normalised_u, z=normalised_v
        )

        depth_map = self.depth_model.predict_depth(cp.frame)
        unscaled_depth: float = depth_map[cp.y_coord, cp.x_coord]

        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi

        x, y, z = self.__spherical_to_cartesian(r=unscaled_depth, theta=theta, phi=phi)

        # relabeling again for the world reference system
        return WorldPoint(frame=cp.frame, x_coord=y, y_coord=z, z_coord=x)

    def __get_world_point(self, cp: CameraPoint) -> WorldPoint:
        unscaled_world_point = self.__get_unscaled_world_point(cp)
        unscaled_world_point.x_coord *= self.scale_factor
        unscaled_world_point.y_coord *= self.scale_factor
        unscaled_world_point.z_coord *= self.scale_factor

        return unscaled_world_point

    def __get_camera_point(self, wp: WorldPoint) -> CameraPoint:
        x, y, z = wp.coords()
        # Note that we here relabel the coordinates to keep the two coordinate systems aligned!
        r, theta, phi = self.__cartesian_to_spherical(x=z, y=x, z=y)
        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi
        z_inner, x_inner, y_inner = self.__spherical_to_cartesian(
            r=np.abs(self.focal_length / (np.sin(theta) * np.cos(phi))),
            theta=theta,
            phi=phi,
        )

        assert np.isclose(z_inner, self.focal_length)

        return CameraPoint(frame=wp.frame, x_coord=x_inner, y_coord=y_inner)

    @staticmethod
    def __spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def __cartesian_to_spherical(x, y, z):
        r = norm([x, y, z])
        if r == 0:
            return 0, 0, 0
        theta = np.arccos(z / r)
        if norm([x, y]) == 0:
            return z, 0, 0
        phi = np.sign(y) * np.arccos(x / norm([x, y]))
        return r, theta, phi

    @staticmethod
    def __calculate_distance_between_world_points(
        wp1: WorldPoint, wp2: WorldPoint
    ) -> float:
        return norm(wp1.coords() - wp2.coords())

    def __get_unscaled_distance_from_camera_points(
        self, cp1: CameraPoint, cp2: CameraPoint
    ):
        unscaled_wp1 = self.__get_unscaled_world_point(cp1)
        unscaled_wp2 = self.__get_unscaled_world_point(cp2)
        return self.__calculate_distance_between_world_points(
            unscaled_wp1, unscaled_wp2
        )


def offline_scaling_factor_estimation_from_least_squares(
    geometric_model: GeometricModel,
    ground_truths: List,
) -> float:
    """Get the scaling factor that should be applied for the speed estimation.

    By applying the least square method to multiple unscaled length predictions this method
    extracts the scaling factor.

    @param geometric_model:
        The GeoMetric model that should be applied to find the scaling factor.
    @param ground_truths:
        The ground truth events that where detected in the video (cars). Each tuple in the list
        holds two points and the distance between those points in meters.
    @return:
        The scaling factor that should be applied for the speed estimation.
    """
    unscaled_predictions = []
    labels = []

    for coords1, coords2, distance in ground_truths:
        f1, u1, v1 = coords1
        f2, u2, v2 = coords2
        cp1 = CameraPoint(frame=f1, x_coord=u1, y_coord=v1)
        cp2 = CameraPoint(frame=f2, x_coord=u2, y_coord=v2)
        unscaled_predictions.append(
            geometric_model.get_unscaled_distance_from_camera_points(cp1, cp2)
        )
        labels.append(distance)

    # return optimal scaling factor under least sum of squares estimator
    return np.dot(unscaled_predictions, labels) / np.dot(
        unscaled_predictions, unscaled_predictions
    )


def __online_scaling_factor_estimation_from_least_squares(stream_of_events):
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
        cp1 = CameraPoint(frame=frame, x_coord=u1, y_coord=v1)
        cp2 = CameraPoint(frame=frame, x_coord=u2, y_coord=v2)
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


def get_ground_truth_events(
    tracking_boxes: Dict[int, List[TrackingBox]]
) -> List[GroundTruthEvent]:
    """Get ground truth events to calculate the scaling factor.

    The method takes tracking boxes as input and derives two points with the corresponding distance
    in meters.

    @param tracking_boxes:
        The TrackingBoxes of the cars that should be analyzed.
    @return:
        Returns a list of GroundTruth events extracted from the TrackingBoxes.
    """
    # extract medium pixel distance traveled by object
    box_distances = []
    for object_id in tracking_boxes:
        start_box = tracking_boxes[object_id][0]
        end_box = tracking_boxes[object_id][-1]
        tracking_box_distance = distance.euclidean(
            [start_box.center_x, start_box.center_y],
            [end_box.center_x, end_box.center_y],
        )
        box_distances.append(tracking_box_distance)

    median_distance = np.percentile(np.array(box_distances), 50)

    # extract ground truth value for each tracking box
    ground_truth_events = []
    for object_id in tracking_boxes:
        center_points = np.array(
            [(box.center_x, box.center_y) for box in tracking_boxes[object_id]]
        )
        start_box = center_points[0]
        end_box = center_points[-1]
        tracking_box_distance = distance.euclidean(start_box, end_box)
        if (
            len(center_points) < 2
            or len(center_points) > 750
            or tracking_box_distance < median_distance
        ):
            continue
        center_points_line = Line(Point(*center_points[0]), Point(*center_points[-1]))

        # extract ground truth value for each tracking box
        for box in tracking_boxes[object_id]:
            # check each of the four lines, spanned by the bounding box rectangle
            upper_line = Line(
                Point(box.x_coord, box.y_coord),
                Point(box.x_coord + box.width, box.y_coord),
            )
            right_line = Line(
                Point(box.x_coord + box.width, box.y_coord),
                Point(box.x_coord + box.width, box.y_coord + box.height),
            )
            lower_line = Line(
                Point(box.x_coord, box.y_coord + box.height),
                Point(box.x_coord + box.width, box.y_coord + box.height),
            )
            left_line = Line(
                Point(box.x_coord, box.y_coord),
                Point(box.x_coord, box.y_coord + box.height),
            )

            intersections = []
            for bounding_box_line in [upper_line, right_line, lower_line, left_line]:
                intersection = get_intersection(center_points_line, bounding_box_line)
                if intersection is not None:
                    intersections.append(intersection)

            if len(intersections) == 2:
                # append ground truth only if line fully cuts bounding box
                intersect1, intersect2 = intersections
                ground_truth_events.append(
                    GroundTruthEvent(
                        (
                            box.frame_count,
                            int(intersect1.x_coord),
                            int(intersect1.y_coord),
                        ),
                        (
                            box.frame_count,
                            int(intersect2.x_coord),
                            int(intersect2.y_coord),
                        ),
                        6,
                    )
                )

    return ground_truth_events
