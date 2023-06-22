import xarray as xr
import numpy as np
import pandas as pd
import unittest

import cartopy.crs as ccrs

import deepsensor.tensorflow as deepsensor

from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # It's safe to share data between tests because the TaskLoader does not modify data
        ds_raw = xr.tutorial.open_dataset("air_temperature")
        self.ds_raw = ds_raw
        self.data_processor = DataProcessor(
            x1_name="lat",
            x2_name="lon",
            x1_map=(ds_raw["lat"].min(), ds_raw["lat"].max()),
            x2_map=(ds_raw["lon"].min(), ds_raw["lon"].max()),
        )
        ds = self.data_processor(ds_raw)
        self.task_loader = TaskLoader(ds, ds)
        self.model = ConvNP(
            self.data_processor,
            self.task_loader,
            unet_channels=(5, 5, 5),
            verbose=False,
        )
        self.task = self.task_loader("2014-12-31")

    def test_receptive_field(self):
        fig = deepsensor.plot.receptive_field(
            self.model.model.receptive_field,
            self.data_processor,
            crs=ccrs.PlateCarree(),
        )

    def test_context_encoding(self):
        fig = deepsensor.plot.context_encoding(self.model, self.task, self.task_loader)

    def test_feature_maps(self):
        figs = deepsensor.plot.feature_maps(self.model, self.task)

    def test_offgrid_context(self):
        mean_ds, std_ds = self.model.predict(self.task, X_t=self.ds_raw)
        fig = mean_ds.isel(time=0).air.plot(cmap="seismic")
        deepsensor.plot.offgrid_context(
            fig.axes, self.task, self.data_processor, self.task_loader
        )
