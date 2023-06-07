import itertools

from parameterized import parameterized


import xarray as xr
import numpy as np
import pandas as pd
import unittest

from tests.utils import gen_random_data_xr, gen_random_data_pandas

from deepsensor.data.loader import TaskLoader


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


class TestTaskLoader(unittest.TestCase):
    """Test TaskLoader

    Tests TODO:
    - Task batching shape as expected
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # It's safe to share data between tests because the TaskLoader does not modify data
        self.da = _gen_data_xr()
        self.df = _gen_data_pandas()

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
            print(repr(tl))
            print(tl)

            for context_sampling, target_sampling in self._gen_task_loader_call_args(
                n_context_and_target, n_context_and_target
            ):
                task = tl("2020-01-01", context_sampling, target_sampling)

        return None

    def test_wrong_length_sampling_strat(self):
        """Sampling strategy must be same length as context/target"""
        tl = TaskLoader(
            context=self.da,
            target=self.da,
        )
        with self.assertRaises(ValueError):
            task = tl("2020-01-01", ["all", "all"], ["all", "all"])

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
