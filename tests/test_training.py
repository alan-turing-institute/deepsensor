import numpy as np
import pandas as pd
import xarray as xr

import unittest

from tqdm import tqdm

import deepsensor.tensorflow as deepsensor

from deepsensor.train.train import Trainer
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP
from deepsensor.data.task import concat_tasks


class TestTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # It's safe to share data between tests because the TaskLoader does not modify data
        ds_raw = xr.tutorial.open_dataset("air_temperature")

        cls.ds_raw = ds_raw
        cls.data_processor = DataProcessor(x1_name="lat", x2_name="lon")

        cls.da = cls.data_processor(ds_raw)

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

    def test_concat_tasks_with_nans(self):
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
            task = tl(date, n_context, 42)
            task["Y_c"][0][:, 0] = np.nan  # Add NaN to context
            task["Y_t"][0][:, 0] = np.nan  # Add NaN to target
            tasks.append(task)

        multiple = 50

        # Check that the context and target data are concatenated correctly
        merged_task = concat_tasks(tasks, multiple=multiple)

        if np.any(np.isnan(merged_task["Y_c"][0].y)):
            raise ValueError("NaNs in the merged context data")

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
            task = tl(date, 10, 10)
            task["Y_c"][0][:, 0] = np.nan  # Add NaN to context
            task["Y_t"][0][:, 0] = np.nan  # Add NaN to target
            print(task)
            train_tasks.append(task)

        # Train
        trainer = Trainer(model, lr=5e-5)
        # batch_size = None
        batch_size = 5
        n_epochs = 10
        epoch_losses = []
        for epoch in tqdm(range(n_epochs)):
            batch_losses = trainer(train_tasks, batch_size=batch_size)
            epoch_losses.append(np.mean(batch_losses))

        # Check for NaNs in the loss
        loss = np.mean(epoch_losses)
        self.assertFalse(np.isnan(loss))

    def test_training_multidim(self):
        """A basic test of the training loop with multidimensional context sets"""
        # Load raw data
        ds_raw = xr.tutorial.open_dataset("air_temperature")

        # Add extra dim
        ds_raw["air2"] = ds_raw["air"].copy()

        # Normalise data
        dp = DataProcessor(x1_name="lat", x2_name="lon")
        ds = dp(ds_raw)

        # Set up task loader
        tl = TaskLoader(context=ds, target=ds)

        # Set up model
        model = ConvNP(dp, tl)

        # Generate training tasks
        n_train_tasks = 10
        train_tasks = []
        for i in range(n_train_tasks):
            date = np.random.choice(self.da.time.values)
            task = tl(date, 10, 10)
            task["Y_c"][0][:, 0] = np.nan  # Add NaN to context
            task["Y_t"][0][:, 0] = np.nan  # Add NaN to target
            print(task)
            train_tasks.append(task)

        # Train
        trainer = Trainer(model, lr=5e-5)
        # batch_size = None
        batch_size = 5
        n_epochs = 10
        epoch_losses = []
        for epoch in tqdm(range(n_epochs)):
            batch_losses = trainer(train_tasks, batch_size=batch_size)
            epoch_losses.append(np.mean(batch_losses))

        # Check for NaNs in the loss
        loss = np.mean(epoch_losses)
        self.assertFalse(np.isnan(loss))
