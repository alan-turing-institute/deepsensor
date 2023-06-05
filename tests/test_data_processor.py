# %%
from copy import deepcopy
from typing import Union

import xarray as xr
import numpy as np
import pandas as pd
import unittest

from deepsensor.data.processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    def _gen_data_xr(self):
        data = np.random.rand(200, 30, 20)
        time = pd.date_range("2020-01-01", "2021-01-01", 200)
        lat = np.linspace(20, 40, 30)
        lon = np.linspace(40, 60, 20)
        da = xr.DataArray(data, coords={"time": time, "x1": lat, "x2": lon})
        return da

    def _gen_data_pandas(self):
        data = np.random.rand(200, 30, 20)

        time = pd.date_range("2020-01-01", "2021-01-01", 200)
        lat = np.linspace(20, 40, 30)
        lon = np.linspace(40, 60, 20)

        mi = pd.MultiIndex.from_product([time, lat, lon], names=["time", "x1", "x2"])
        df = pd.DataFrame(data.flatten(), index=mi, columns=["t2m"])
        return df

    def assert_allclose_pd(
        self, df1: Union[pd.DataFrame, pd.Series], df2: Union[pd.DataFrame, pd.Series]
    ):
        if isinstance(df1, pd.Series):
            df1 = df1.to_frame()
        try:
            pd.testing.assert_frame_equal(df1, df2)
        except AssertionError:
            return False
        return True

    def assert_allclose_xr(
        self, da1: Union[xr.DataArray, xr.Dataset], da2: Union[xr.DataArray, xr.Dataset]
    ):
        try:
            xr.testing.assert_allclose(da1, da2)
        except AssertionError:
            return False
        return True

    def test_same_names_xr(self):
        da = self._gen_data_xr()

        original_da = deepcopy(da)
        dp = DataProcessor(x1_map=(20, 40), x2_map=(40, 60), x1_name="x1", x2_name="x2")
        da = dp(da)

        self.assertListEqual(["time", "x1", "x2"], list(da.dims))

        da = dp.unnormalise(da)

        self.assertTrue(
            self.assert_allclose_xr(da, original_da), f"Original {type(da).__name__} not restored."
        )

    def test_different_names_xr(self):
        da = self._gen_data_xr()
        da = da.rename({"time": "datetime", "x1": "latitude", "x2": "longitude"})
        original_da = deepcopy(da)
        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="datetime",
            x1_name="latitude",
            x2_name="longitude",
        )
        da = dp(da)
        self.assertListEqual(
            ["time", "x1", "x2"], list(da.dims), "Failed to rename dims."
        )

        da = dp.unnormalise(da)
        self.assertTrue(
            self.assert_allclose_xr(da, original_da), f"Original {type(da).__name__} not restored."
        )

    def test_wrong_order_xr(self):
        da = self._gen_data_xr()
        # Transpose, changing order
        da = da.T
        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="x1",
            x2_name="x2",
        )
        with self.assertRaises(ValueError):
            dp(da)

    def test_same_names_pandas(self):
        df = self._gen_data_pandas()
        original_df = deepcopy(df)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="x1",
            x2_name="x2",
        )

        df = dp(df)

        self.assertListEqual(["time", "x1", "x2"], list(df.index.names))

        df = dp.unnormalise(df)

        self.assertTrue(
            self.assert_allclose_pd(df, original_df), f"Original {type(df).__name__} not restored."
        )

    def test_different_names_pandas(self):
        df = self._gen_data_pandas()
        original_df = deepcopy(df)

        df.index.names = ["datetime", "lat", "lon"]

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="datetime",
            x1_name="lat",
            x2_name="lon",
        )

        df = dp(df)

        self.assertListEqual(["time", "x1", "x2"], list(df.index.names))

        df = dp.unnormalise(df)

        self.assertTrue(
            self.assert_allclose_pd(df, original_df), f"Original {type(df).__name__} not restored."
        )

    def test_wrong_order_pandas(self):
        df = self._gen_data_pandas()

        df = df.swaplevel(0, 2)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )

        with self.assertRaises(ValueError):
            dp(df)


if __name__ == "__main__":
    unittest.main()
