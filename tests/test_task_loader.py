import itertools

from parameterized import parameterized

import xarray as xr
import dask.array
import numpy as np
import pandas as pd
import unittest

from deepsensor.errors import InvalidSamplingStrategyError
from tests.utils import gen_random_data_xr, gen_random_data_pandas

from deepsensor.data.loader import TaskLoader


def _gen_data_xr(coords=None, dims=None, data_vars=None, use_dask=False):
    """Gen random normalised data"""
    if coords is None:
        coords = dict(
            time=pd.date_range("2020-01-01", "2020-01-31", freq="D"),
            x1=np.linspace(0, 1, 30),
            x2=np.linspace(0, 1, 20),
        )
    da = gen_random_data_xr(coords, dims, data_vars)
    if use_dask:
        da.data = dask.array.from_array(da.data)
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


class TestTaskLoader(unittest.TestCase):
    """Test TaskLoader

    Tests TODO:
    - Task batching shape as expected
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # It's safe to share data between tests because the TaskLoader does not modify data
        self.da = _gen_data_xr()
        self.aux_da = self.da.isel(time=0)
        self.df = _gen_data_pandas()

    def _gen_task_loader_call_args(self, n_context_sets, n_target_sets):
        """Generate arguments for TaskLoader.__call__

        Loops over all possible combinations of context/target sampling methods
        and returns a list of arguments for TaskLoader.__call__.
        Options tested include:
        - (int): Random number of samples
        - (float): Fraction of samples
        - "all": All samples
        - (np.ndarray): Array of coordinates to sample from the dataset

        Args:
            n_context_sets (int): Number of context samples
            n_target_sets (int): Number of target samples
        Returns:
            (tuple): Arguments for TaskLoader.__call__
        """
        for sampling_method in [
            0.0,
            0,
            10,
            0.5,
            "all",
            np.zeros((2, 1)),
        ]:
            yield [sampling_method] * n_context_sets, [sampling_method] * n_target_sets

    def test_load_dask(self):
        """Test loading dask data"""
        da = _gen_data_xr(use_dask=True)
        aux_da = da.isel(time=0)
        tl = TaskLoader(
            context=da, target=da, aux_at_targets=aux_da, aux_at_contexts=aux_da
        )
        tl.load_dask()

    @parameterized.expand(range(1, 4))
    def test_loader_call(self, n_context_and_target):
        """Test TaskLoader.__call__ for all possible combinations of context/target sampling methods

        Generates all possible combinations of xarray and pandas context/target sets
        of length n_context_and_target and calls TaskLoader.__call__ with all possible sampling methods.

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

        def data_type_ID_to_data(set_list):
            """Converts a list of data type IDs ("pd" or "xr") to a list of data objects of that type

            E.g. ["xr", "pd", "xr"] -> [self.da, self.df, self.da]
            E.g. "xr" -> self.da
            """
            if set_list == "xr":
                return self.da
            elif set_list == "pd":
                return self.df
            elif isinstance(set_list, (list, tuple)):
                return [data_type_ID_to_data(s) for s in set_list]

        for context_IDs, target_IDs in zip(context_ID_list, target_ID_list):
            context = data_type_ID_to_data(context_IDs)
            target = data_type_ID_to_data(target_IDs)
            tl = TaskLoader(context=context, target=target)

            for context_sampling, target_sampling in self._gen_task_loader_call_args(
                n_context_and_target, n_context_and_target
            ):
                task = tl("2020-01-01", context_sampling, target_sampling)

        return None

    def test_aux_at_contexts_and_aux_at_targets(self):
        """Test the `aux_at_contexts` and `aux_at_targets` arguments"""
        context = [self.da, self.df]
        target = self.df

        tl = TaskLoader(
            context=[self.da, self.df],  # gridded xarray and off-grid pandas contexts
            target=self.df,  # off-grid pandas targets
            aux_at_contexts=self.aux_da,  # gridded xarray to sample at off-grid context locations
            aux_at_targets=self.aux_da,  # gridded xarray to sample at target locations
        )

        for context_sampling, target_sampling in self._gen_task_loader_call_args(
            len(context), 1
        ):
            task = tl("2020-01-01", context_sampling, target_sampling)

    def test_invalid_sampling_strat(self):
        """Test invalid sampling strategy in `TaskLoader.__call__`

        Here we only need to test context sampling strategies because the same code to check
        the validity of the sampling strategy is used for context and target sampling.
        """
        invalid_context_sampling_strategies = [
            # Sampling strategy must be same length as context/target
            ["all", "all", "all"],
            # If integer, sampling strategy must be positive
            -1,
            # If float, sampling strategy must be less than or equal to 1.0
            1.1,
            # If float, sampling strategy must be greater than or equal to 0.0
            -0.1,
            # If str, sampling strategy must be in ["all", "split"]
            "invalid",
            # If np.ndarray, sampling strategy must be shape (2, N)
            np.zeros((1, 2, 2)),
            # If np.ndarray, coordinates must exist in the dataset
            np.ones((2, 1)) * 1000,
            # Invalid type
            dict(foo="bar"),
        ]

        for tl in [
            TaskLoader(
                context=self.da,
                target=self.da,
            ),
            TaskLoader(
                context=self.df,
                target=self.df,
            ),
        ]:
            for invalid_sampling_strategy in invalid_context_sampling_strategies:
                with self.assertRaises(InvalidSamplingStrategyError):
                    task = tl("2020-01-01", invalid_sampling_strategy)

    def test_split_fails_if_not_df(self):
        """The "split" sampling strategy only works with pandas objects (currently)"""
        with self.assertRaises(ValueError):
            # Indexes don't connect two pandas objects
            tl = TaskLoader(context=self.df, target=self.da, links=[(0, 0)])

    def test_wrong_links(self):
        """Test link indexes out of range"""
        with self.assertRaises(ValueError):
            tl = TaskLoader(context=self.df, target=self.df, links=[(0, 1)])

    def test_links(self):
        """Test sampling from linked dataframes works as expected"""
        tl = TaskLoader(context=self.df, target=self.df, links=[(0, 0)])
        task = tl("2020-01-01", "split", "split", split_frac=0.0)
        self.assertEqual(task["Y_c"][0].size, 0)  # Should be no context data
        task = tl("2020-01-01", "split", "split", split_frac=1.0)
        self.assertEqual(task["Y_t"][0].size, 0)  # Should be no target data
        task = tl("2020-01-01", "split", "split", split_frac=0.5)
        self.assertEqual(
            task["Y_c"][0].size // 2, task["Y_t"][0].size // 2
        )  # Should be split equally

        # Should raise ValueError if "split" provided for context but not target (or vice versa)
        with self.assertRaises(ValueError):
            task = tl("2020-01-01", "split", "all")
        with self.assertRaises(ValueError):
            task = tl("2020-01-01", "all", "split")

        # Should raise ValueError if `split_frac` not between 0 and 1
        with self.assertRaises(ValueError):
            task = tl("2020-01-01", "split", "split", split_frac=1.1)
            task = tl("2020-01-01", "split", "split", split_frac=-0.1)


if __name__ == "__main__":
    unittest.main()
