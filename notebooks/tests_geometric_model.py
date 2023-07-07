import unittest
from itertools import product

import numpy as np

from notebooks.scaling_factor_extraction import (
    DepthModel,
    GeometricModel,
    WorldPoint,
    CameraPoint,
)


class GeometricModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depth_model = DepthModel()
        self.geometric_model = GeometricModel(self.depth_model)

    def test_coordinate_translations(self):
        coords = [-1, 1]
        # going through the eight quadrants
        for point in product(coords, repeat=3):
            r, theta, phi = self.geometric_model._cartesian_to_spherical(*point)
            self.assertEqual(np.sqrt(3), r)
            new_point = self.geometric_model._spherical_to_cartesian(r, theta, phi)
            self.assertTrue(np.allclose(new_point, point))

    def test_world_to_camera_translation(self):
        frame = np.empty((100, 100))

        # point right in front of camera
        wp = WorldPoint(frame, 0, 0, 1)
        cp = self.geometric_model.get_camera_point(wp)
        self.assertTrue(np.allclose(cp.coords(), [0, 0]))

        coords = [-1, 1]
        # going through the four quadrants in front of the camera
        for (x, y) in product(coords, repeat=2):
            wp = WorldPoint(frame, x, y, 1)
            cp = self.geometric_model.get_camera_point(wp)
            # we expect that due to point always appears in the diagonally opposite quadrant of the camera ref system
            self.assertTrue(np.allclose(cp.coords(), [-x, -y]))

    def test_camera_to_world_translation(self):
        # the depth model will predict a depth of sqrt(3) for all points.
        frame = np.full((3, 3), np.sqrt(3))
        self.geometric_model.c_u = 1
        self.geometric_model.c_v = 1

        # origin of screen
        cp = CameraPoint(frame, 1, 1)
        wp = self.geometric_model.get_world_point(cp)
        self.assertTrue(np.allclose(wp.coords(), [0, 0, np.sqrt(3)]))

        coords = [0, 2]
        # going through the four quadrants of the screen
        for (u, v) in product(coords, repeat=2):
            cp = CameraPoint(frame, u, v)
            wp = self.geometric_model.get_world_point(cp)
            # we expect that due to point always appears in the diagonally opposite quadrant of the camera ref system
            self.assertTrue(np.allclose(wp.coords(), [-(u - 1), -(v - 1), 1]))

    def test_distance_between_world_points(self):
        frame = np.empty((100, 100))
        wp1 = WorldPoint(frame, 0, 0, 1)
        wp2 = WorldPoint(frame, 0, 0, 0)
        self.assertEqual(
            1, self.geometric_model.calculate_distance_between_world_points(wp1, wp2)
        )

        wp1 = WorldPoint(frame, 1, 1, 1)
        self.assertTrue(
            np.isclose(
                np.sqrt(3),
                self.geometric_model.calculate_distance_between_world_points(wp1, wp2),
            )
        )

    def test_distance_from_camera_points(self):
        # the depth model will predict a depth of sqrt(3) for all points.
        frame = np.full((3, 3), np.sqrt(3))
        self.geometric_model.c_u = 1
        self.geometric_model.c_v = 1

        # points should be projected onto different corners of the cube of side length 2 cenetered at origin
        cp1 = CameraPoint(frame, 2, 2)
        cp2 = CameraPoint(frame, 0, 0)
        distance = self.geometric_model.get_distance_from_camera_points(cp1, cp2)
        self.assertTrue(np.isclose(distance, 2 * np.sqrt(2)))

        cp2 = CameraPoint(frame, 2, 0)
        distance = self.geometric_model.get_distance_from_camera_points(cp1, cp2)
        self.assertTrue(np.isclose(distance, 2))


if __name__ == "__main__":
    unittest.main()
