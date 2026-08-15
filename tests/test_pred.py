import unittest

import numpy as np
import xarray as xr

from deepsensor.model.pred import increase_spatial_resolution


class TestPredictionHelpers(unittest.TestCase):
    def test_increase_spatial_resolution_with_xarray_coords(self):
        data = xr.DataArray(
            np.arange(24).reshape(4, 6),
            coords={
                "x1": np.linspace(0.0, 1.0, 4),
                "x2": np.linspace(10.0, 20.0, 6),
            },
            dims=("x1", "x2"),
        )

        for xarray_obj in (data, data.to_dataset(name="value")):
            with self.subTest(type=type(xarray_obj).__name__):
                result = increase_spatial_resolution(xarray_obj, 0.5)

                self.assertEqual(result.sizes["x1"], 2)
                self.assertEqual(result.sizes["x2"], 3)
                np.testing.assert_allclose(result.x1, [0.0, 1.0])
                np.testing.assert_allclose(result.x2, [10.0, 15.0, 20.0])


if __name__ == "__main__":
    unittest.main()
