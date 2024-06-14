import unittest

import numpy as np
import xarray as xr

from deepsensor import DataProcessor, TaskLoader
from deepsensor.data.task import append_obs_to_task
from deepsensor.errors import TaskSetIndexError, GriddedDataError
from deepsensor.model import ConvNP


class TestConcatTasks(unittest.TestCase):

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

    def test_concat_obs_to_task_shapes(self):
        ctx_idx = 0  # Context set index to add new observations to

        # Sample 10 context observations
        task = self.task_loader("2014-12-31", context_sampling=10)

        # 1 context observation
        X_new = np.random.randn(2, 1)
        Y_new = np.random.randn(1, 1)
        new_task = append_obs_to_task(task, X_new, Y_new, ctx_idx)
        self.assertEqual(new_task["X_c"][ctx_idx].shape, (2, 11))
        self.assertEqual(new_task["Y_c"][ctx_idx].shape, (1, 11))

        # 1 context observation with flattened obs dim
        X_new = np.random.randn(2)
        Y_new = np.random.randn(1)
        new_task = append_obs_to_task(task, X_new, Y_new, ctx_idx)
        self.assertEqual(new_task["X_c"][ctx_idx].shape, (2, 11))
        self.assertEqual(new_task["Y_c"][ctx_idx].shape, (1, 11))

        # 5 context observations
        X_new = np.random.randn(2, 5)
        Y_new = np.random.randn(1, 5)
        new_task = append_obs_to_task(task, X_new, Y_new, ctx_idx)
        self.assertEqual(new_task["X_c"][ctx_idx].shape, (2, 15))
        self.assertEqual(new_task["Y_c"][ctx_idx].shape, (1, 15))

    def test_concat_obs_to_task_wrong_context_index(self):
        # Sample 10 context observations
        task = self.task_loader("2014-12-31", context_sampling=10)

        ctx_idx = 1  # Wrong context set index

        # 1 context observation
        X_new = np.random.randn(2, 1)
        Y_new = np.random.randn(1, 1)

        with self.assertRaises(TaskSetIndexError):
            _ = append_obs_to_task(task, X_new, Y_new, ctx_idx)

    def test_concat_obs_to_task_fails_for_gridded_data(self):
        ctx_idx = 0  # Context set index to add new observations to

        # Sample context observations on a grid
        task = self.task_loader("2014-12-31", context_sampling="all")

        # Confirm that context observations are gridded with tuple for coordinates
        assert isinstance(task["X_c"][ctx_idx], tuple)

        # 1 context observation
        X_new = np.random.randn(2, 1)
        Y_new = np.random.randn(1, 1)

        with self.assertRaises(GriddedDataError):
            new_task = append_obs_to_task(task, X_new, Y_new, ctx_idx)
