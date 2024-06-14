import xarray as xr
import numpy as np
import pandas as pd
import unittest

import deepsensor.tensorflow as deepsensor

from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP


class TestPlotting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # It's safe to share data between tests because the TaskLoader does not modify data
        ds_raw = xr.tutorial.open_dataset("air_temperature")
        cls.ds_raw = ds_raw
        cls.data_processor = DataProcessor(x1_name="lat", x2_name="lon")
        ds = cls.data_processor(ds_raw)
        cls.task_loader = TaskLoader(context=ds, target=ds)
        cls.model = ConvNP(
            cls.data_processor,
            cls.task_loader,
            unet_channels=(5, 5, 5),
            verbose=False,
        )
        # Sample a task with 10 random context points
        cls.task = cls.task_loader(
            "2014-12-31", context_sampling=10, target_sampling="all"
        )

    def test_context_encoding(self):
        fig = deepsensor.plot.context_encoding(self.model, self.task, self.task_loader)

    def test_feature_maps(self):
        figs = deepsensor.plot.feature_maps(self.model, self.task)

    def test_offgrid_context(self):
        pred = self.model.predict(self.task, X_t=self.ds_raw)
        fig = pred["air"]["mean"].isel(time=0).plot(cmap="seismic")
        deepsensor.plot.offgrid_context(
            fig.axes, self.task, self.data_processor, self.task_loader
        )

    def test_offgrid_context_observations(self):
        pred = self.model.predict(self.task, X_t=self.ds_raw)
        fig = pred["air"]["mean"].isel(time=0).plot(cmap="seismic")
        deepsensor.plot.offgrid_context_observations(
            fig.axes,
            self.task,
            self.data_processor,
            self.task_loader,
            context_set_idx=0,
            format_str=None,
            extent=None,
            color="black",
        )
