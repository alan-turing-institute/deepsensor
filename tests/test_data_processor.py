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
        da_raw = self._gen_data_xr()

        da_raw = deepcopy(da_raw)
        dp = DataProcessor(x1_map=(20, 40), x2_map=(40, 60), x1_name="x1", x2_name="x2")
        da_norm = dp(da_raw)

        self.assertListEqual(["time", "x1", "x2"], list(da_norm.dims))

        da_unnorm = dp.unnormalise(da_norm)

        self.assertTrue(
            self.assert_allclose_xr(da_unnorm, da_raw), f"Original {type(da_raw).__name__} not restored."
        )

    def test_different_names_xr(self):
        da_raw = self._gen_data_xr()
        da_raw = da_raw.rename({"time": "datetime", "x1": "latitude", "x2": "longitude"})
        da_raw = deepcopy(da_raw)
        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="datetime",
            x1_name="latitude",
            x2_name="longitude",
        )
        da_norm = dp(da_raw)
        self.assertListEqual(
            ["time", "x1", "x2"], list(da_norm.dims), "Failed to rename dims."
        )

        da_unnorm = dp.unnormalise(da_norm)
        self.assertTrue(
            self.assert_allclose_xr(da_unnorm, da_raw), f"Original {type(da_raw).__name__} not restored."
        )

    def test_wrong_order_xr(self):
        da_raw = self._gen_data_xr()
        # Transpose, changing order
        da_raw = da_raw.T
        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="x1",
            x2_name="x2",
        )
        with self.assertRaises(ValueError):
            dp(da_raw)

    def test_same_names_pandas(self):
        df_raw = self._gen_data_pandas()
        df_raw = deepcopy(df_raw)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="x1",
            x2_name="x2",
        )

        df_norm = dp(df_raw)

        self.assertListEqual(["time", "x1", "x2"], list(df_norm.index.names))

        df_unnorm = dp.unnormalise(df_norm)

        self.assertTrue(
            self.assert_allclose_pd(df_unnorm, df_raw), f"Original {type(df_raw).__name__} not restored."
        )

    def test_different_names_pandas(self):
        df_raw = self._gen_data_pandas()
        df_raw = deepcopy(df_raw)

        df_raw.index.names = ["datetime", "lat", "lon"]

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="datetime",
            x1_name="lat",
            x2_name="lon",
        )

        df_norm = dp(df_raw)

        self.assertListEqual(["time", "x1", "x2"], list(df_norm.index.names))

        df_unnorm = dp.unnormalise(df_norm)

        self.assertTrue(
            self.assert_allclose_pd(df_unnorm, df_raw), f"Original {type(df_raw).__name__} not restored."
        )

    def test_wrong_order_pandas(self):
        df_raw = self._gen_data_pandas()

        df_raw = df_raw.swaplevel(0, 2)

        dp = DataProcessor(
            x1_map=(20, 40),
            x2_map=(40, 60),
            time_name="time",
            x1_name="lat",
            x2_name="lon",
        )

        with self.assertRaises(ValueError):
            dp(df_raw)


if __name__ == "__main__":
    unittest.main()
