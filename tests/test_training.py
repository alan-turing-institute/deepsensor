import numpy as np
import pandas as pd
import xarray as xr

import unittest

from tqdm import tqdm

import deepsensor.tensorflow as deepsensor

from deepsensor.train.train import train_epoch
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP, concat_tasks

from tests.utils import gen_random_data_xr, gen_random_data_pandas


class TestTraining(unittest.TestCase):
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

        self.da = self.data_processor(ds_raw)

    def test_concat_tasks(self):
        tl = TaskLoader(context=self.da, target=self.da)

        seed = 42
        rng = np.random.default_rng(seed)

        n_tasks = 5
        tasks = []
        tasks_different_n_targets = []
        for i in range(n_tasks):
            n_context = rng.integers(1, 100)
            n_target = rng.integers(1, 100)
            date = rng.choice(self.da.time.values)
            tasks_different_n_targets.append(
                tl(date, n_context, n_target)
            )  # Changing number of targets
            tasks.append(tl(date, n_context, 42))  # Fixed number of targets

        multiple = 50
        with self.assertRaises(ValueError):
            merged_task = concat_tasks(tasks_different_n_targets, multiple=multiple)

        # Check that the context and target data are concatenated correctly
        merged_task = concat_tasks(tasks, multiple=multiple)

    def test_training(self):
        """A basic test of the training loop

        Note: This could be extended into a regression test, e.g. checking the loss decreases,
        the model parameters change, the speed of training is reasonable, etc.
        """
        tl = TaskLoader(context=self.da, target=self.da)
        model = ConvNP(self.data_processor, tl, unet_channels=(5, 5, 5), verbose=False)

        # Generate training tasks
        n_train_tasks = 10
        train_tasks = []
        for i in range(n_train_tasks):
            date = np.random.choice(self.da.time.values)
            train_tasks.append(tl(date, 10, 10))

        # Train
        # batch_size = None
        batch_size = 5
        n_epochs = 10
        epoch_losses = []
        for epoch in tqdm(range(n_epochs)):
            batch_losses = train_epoch(model, train_tasks, batch_size=batch_size)
            epoch_losses.append(np.mean(batch_losses))

        # Check for NaNs in the loss
        loss = np.mean(epoch_losses)
        self.assertFalse(np.isnan(loss))
