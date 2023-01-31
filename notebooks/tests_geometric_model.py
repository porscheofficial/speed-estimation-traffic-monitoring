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

            r, theta, phi = self.geometric_model._cartesian_to_sperhical(*point)
            self.assertEqual(np.sqrt(3), r)
            new_point = self.geometric_model._spherical_to_cartesian(r, theta, phi)
            self.assertTrue(np.allclose(new_point, point))


if __name__ == "__main__":
    unittest.main()
