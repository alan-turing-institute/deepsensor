import copy
import itertools

from parameterized import parameterized


import xarray as xr
import numpy as np
import pandas as pd
import unittest

import lab as B

import deepsensor.tensorflow as deepsensor

from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP

from tests.utils import gen_random_data_xr, gen_random_data_pandas


def _gen_data_xr(coords=None, dims=None, data_vars=None):
    """Gen random normalised data"""
    if coords is None:
        coords = dict(
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            x1=np.linspace(0, 1, 30),
            x2=np.linspace(0, 1, 20),
        )
    da = gen_random_data_xr(coords, dims, data_vars)
    return da


def _gen_data_pandas(coords=None, dims=None, cols=None):
    """Gen random normalised data"""
    if coords is None:
        coords = dict(
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            x1=np.linspace(0, 1, 10),
            x2=np.linspace(0, 1, 10),
        )
    df = gen_random_data_pandas(coords, dims, cols)
    return df


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # It's safe to share data between tests because the TaskLoader does not modify data
        self.da = _gen_data_xr()
        self.df = _gen_data_pandas()

        self.dp = DataProcessor()
        _ = self.dp([self.da, self.df])  # Compute normalization parameters

    def _gen_task_loader_call_args(self, n_context, n_target):
        """Generate arguments for TaskLoader.__call__

        Loops over all possible combinations of context/target sampling methods
        and returns a list of arguments for TaskLoader.__call__.
        Options tested include:
        - (int): Random number of samples
        - (float): Fraction of samples
        - "all": All samples

        Args:
            n_context (int): Number of context samples
            n_target (int): Number of target samples
        Returns:
            (tuple): Arguments for TaskLoader.__call__
        """
        for sampling_method in [
            10,
            0.5,
            "all",
        ]:
            yield [sampling_method] * n_context, [sampling_method] * n_target

    # TEMP only 1D because non-overlapping target sets are not yet supported
    @parameterized.expand(range(1, 2))
    def test_model_call(self, n_context_and_target):
        """Check `ConvNP` runs with all possible combinations of context/target sampling methods

        Generates all possible combinations of xarray and pandas context/target sets
        of length n_context_and_target runs `ConvNP` with all possible combinations of
        context/target sampling methods.

        Args:
            n_context_and_target (int): Number of context and target sets
        """
        # Convert to list of strings containing every possible combination of "xr" and "pd"
        context_ID_list = list(
            itertools.product(["xr", "pd"], repeat=n_context_and_target)
        )
        target_ID_list = list(
            itertools.product(["xr", "pd"], repeat=n_context_and_target)
        )

        def set_list_to_data(set_list):
            if set_list == "xr":
                return self.da
            elif set_list == "pd":
                return self.df
            elif isinstance(set_list, (list, tuple)):
                return [set_list_to_data(s) for s in set_list]

        for context_IDs, target_IDs in zip(context_ID_list, target_ID_list):
            tl = TaskLoader(
                context=set_list_to_data(context_IDs),
                target=set_list_to_data(target_IDs),
            )

            model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)

            for context_sampling, target_sampling in self._gen_task_loader_call_args(
                n_context_and_target, n_context_and_target
            ):
                task = tl("2020-01-01", context_sampling, target_sampling)
                dist = model(task)

        return None

    @parameterized.expand(range(1, 4))
    def test_prediction_shapes_lowlevel(self, target_dim):
        """Test low-level model prediction interface"""
        tl = TaskLoader(
            context=self.da,
            target=[self.da] * target_dim,
        )

        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)

        context_sampling = 10
        for expected_shape, target_sampling in (
            ((10,), 10),
            (self.da.shape[-2:], "all"),
        ):
            task = tl("2020-01-01", context_sampling, target_sampling)

            dist = model(task)

            n_targets = np.product(expected_shape)

            # Tensors
            assert_shape(model.mean(dist), (target_dim, *expected_shape))
            assert_shape(model.mean(task), (target_dim, *expected_shape))
            assert_shape(model.variance(dist), (target_dim, *expected_shape))
            assert_shape(model.variance(task), (target_dim, *expected_shape))
            assert_shape(model.stddev(dist), (target_dim, *expected_shape))
            assert_shape(model.stddev(task), (target_dim, *expected_shape))
            assert_shape(
                model.covariance(dist), (n_targets * target_dim, n_targets * target_dim)
            )
            assert_shape(
                model.covariance(task), (n_targets * target_dim, n_targets * target_dim)
            )
            n_samples = 5
            assert_shape(
                model.sample(dist, n_samples), (n_samples, target_dim, *expected_shape)
            )
            assert_shape(
                model.sample(task, n_samples), (n_samples, target_dim, *expected_shape)
            )

            # Scalars
            x = model.logpdf(dist, task)
            assert x.size == 1 and x.shape == ()
            x = model.logpdf(task)
            assert x.size == 1 and x.shape == ()
            x = model.entropy(dist)
            assert x.size == 1 and x.shape == ()
            x = model.entropy(task)
            assert x.size == 1 and x.shape == ()
            x = B.to_numpy(model.loss_fn(task))
            assert x.size == 1 and x.shape == ()

    @parameterized.expand(range(1, 4))
    def test_prediction_shapes_highlevel(self, target_dim):
        """Test high-level `.predict` interface"""

        if target_dim > 1:
            # Avoid data var name clash in `predict`
            target_names = [f"target_{i}" for i in range(target_dim)]
            target = [self.da] * target_dim
            for i, name in enumerate(target_names):
                target[i] = copy.deepcopy(target[i])
                target[i].name = name
        elif target_dim == 1:
            target = self.da
        tl = TaskLoader(
            context=self.da,
            target=target,
        )

        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)

        dates = pd.date_range("2020-01-01", "2020-01-07", freq="D")
        tasks = [tl(date) for date in dates]

        # Gridded predictions
        n_samples = 5
        mean_ds, std_ds, samples_ds = model.predict(
            tasks,
            X_t=self.da,
            n_samples=n_samples,
            unnormalise=False if target_dim > 1 else True,
        )
        assert [isinstance(ds, xr.Dataset) for ds in [mean_ds, std_ds, samples_ds]]
        assert_shape(
            mean_ds.to_array(),
            (target_dim, len(dates), self.da.x1.size, self.da.x2.size),
        )
        assert_shape(
            std_ds.to_array(),
            (target_dim, len(dates), self.da.x1.size, self.da.x2.size),
        )
        assert_shape(
            samples_ds.to_array(),
            (target_dim, n_samples, len(dates), self.da.x1.size, self.da.x2.size),
        )

        # Offgrid predictions
        n_samples = 5
        X_t = self.df.loc[dates[0]]
        mean_df, std_df, samples_df = model.predict(
            tasks,
            X_t=X_t,
            n_samples=n_samples,
            unnormalise=False if target_dim > 1 else True,
        )
        assert [isinstance(df, pd.DataFrame) for df in [mean_df, std_df, samples_df]]
        n_preds = len(dates) * len(X_t)
        assert_shape(mean_df, (n_preds, target_dim))
        assert_shape(std_df, (n_preds, target_dim))
        assert_shape(samples_df, (n_samples * n_preds, target_dim))


def assert_shape(x, shape: tuple):
    """ex: assert_shape(conv_input_array, [8, 3, None, None])"""
    # TODO put this in a utils module?
    assert len(x.shape) == len(shape), (x.shape, shape)
    for _a, _b in zip(x.shape, shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, shape)
