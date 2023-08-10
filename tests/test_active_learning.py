import unittest
import xarray as xr
import numpy as np

import deepsensor.tensorflow as deepsensor
from deepsensor.active_learning.acquisition_fns import (
    MeanVariance,
    MeanStddev,
    pNormStddev,
    MeanMarginalEntropy,
    JointEntropy,
    Stddev,
    ExpectedImprovement,
    Random,
    OracleMAE,
    OracleRMSE,
    OracleMarginalNLL,
    OracleJointNLL,
    AcquisitionFunction,
)
from deepsensor.active_learning.algorithms import GreedyAlgorithm

from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor, xarray_to_coord_array_normalised
from deepsensor.data.task import append_obs_to_task
from deepsensor.errors import TaskSetIndexError, GriddedDataError
from deepsensor.model.convnp import ConvNP


# from deepsensor.active_learning.acquisition_fns import


class TestConcatTasks(unittest.TestCase):
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
        self.task_loader = TaskLoader(context=ds, target=ds)
        self.model = ConvNP(
            self.data_processor,
            self.task_loader,
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


class TestActiveLearning(unittest.TestCase):
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
        self.ds = self.data_processor(ds_raw)
        self.task_loader = TaskLoader(context=self.ds, target=self.ds)
        self.model = ConvNP(
            self.data_processor,
            self.task_loader,
            unet_channels=(5, 5, 5),
            verbose=False,
        )

        # Set up model with aux-at-target data
        aux_at_targets = self.ds.isel(time=0).drop_vars("time")
        self.task_loader_with_aux = TaskLoader(
            context=self.ds, target=self.ds, aux_at_targets=aux_at_targets
        )
        self.model_with_aux = ConvNP(
            self.data_processor,
            self.task_loader_with_aux,
            unet_channels=(5, 5, 5),
            verbose=False,
        )

    def test_wrong_n_new_sensors(self):
        with self.assertRaises(ValueError):
            alg = GreedyAlgorithm(
                model=self.model,
                X_t=self.ds_raw,
                X_s=self.ds_raw,
                N_new_context=-1,
            )

        with self.assertRaises(ValueError):
            alg = GreedyAlgorithm(
                model=self.model,
                X_t=self.ds_raw,
                X_s=self.ds_raw,
                N_new_context=10_000,  # > number of search points
            )

    def test_acquisition_fns_run(self):
        """Run each acquisition function to check that it runs and returns correct shape"""
        sequential_acquisition_fns = [
            MeanStddev(self.model),
            MeanVariance(self.model),
            pNormStddev(self.model, p=3),
            MeanMarginalEntropy(self.model),
            JointEntropy(self.model),
            OracleMAE(self.model),
            OracleRMSE(self.model),
            OracleMarginalNLL(self.model),
            OracleJointNLL(self.model),
        ]
        parallel_acquisition_fns = [
            Stddev(self.model),
            ExpectedImprovement(self.model),
            Random(),
        ]

        # Coarsen search points to speed up computation
        X_s = self.ds_raw.air.coarsen(lat=10, lon=10, boundary="trim").mean()
        X_s = self.data_processor.map_coords(X_s)
        X_s_arr = xarray_to_coord_array_normalised(X_s)

        task = self.task_loader("2014-12-31", context_sampling=10)

        for acquisition_fn in sequential_acquisition_fns:
            importance = acquisition_fn(task)
            assert importance.size == 1
        for acquisition_fn in parallel_acquisition_fns:
            importances = acquisition_fn(task, X_s_arr)
            assert importances.size == X_s_arr.shape[-1]

    def test_greedy_alg_runs(self):
        """Run the greedy algorithm to check that it runs without error"""
        # Both a sequential and parallel acquisition function
        acquisition_fns = [
            MeanStddev(self.model),
            Stddev(self.model),
        ]

        # Coarsen search points to speed up computation
        X_s = self.ds_raw.air.coarsen(lat=10, lon=10, boundary="trim").mean()

        alg = GreedyAlgorithm(
            model=self.model,
            X_t=X_s,
            X_s=X_s,
            N_new_context=2,
        )

        task = self.task_loader("2014-12-31", context_sampling=10)

        for acquisition_fn in acquisition_fns:
            X_new_df, acquisition_fn_ds = alg(acquisition_fn, task)

    def test_greedy_alg_with_aux_at_targets(self):
        """Run the greedy algorithm to check that it runs without error"""
        # Both a sequential and parallel acquisition function
        acquisition_fns = [
            MeanStddev(self.model_with_aux),
            Stddev(self.model_with_aux),
        ]

        # Coarsen search points to speed up computation
        X_s = self.ds_raw.air.coarsen(lat=10, lon=10, boundary="trim").mean()

        alg = GreedyAlgorithm(
            model=self.model_with_aux,
            X_t=X_s,
            X_s=X_s,
            N_new_context=2,
            task_loader=self.task_loader_with_aux,
        )

        task = self.task_loader_with_aux("2014-12-31", context_sampling=10)

        for acquisition_fn in acquisition_fns:
            X_new_df, acquisition_fn_ds = alg(acquisition_fn, task)

    def test_greedy_alg_with_oracle_acquisition_fn(self):
        acquisition_fn = OracleMAE(self.model)

        # Coarsen search points to speed up computation
        X_s = self.ds_raw.air.coarsen(lat=10, lon=10, boundary="trim").mean()

        alg = GreedyAlgorithm(
            model=self.model,
            X_t=X_s,
            X_s=X_s,
            N_new_context=2,
            task_loader=self.task_loader,
        )

        task = self.task_loader("2014-12-31", context_sampling=10)

        _ = alg(acquisition_fn, task)

    def test_greedy_alg_with_sequential_acquisition_fn(self):
        acquisition_fn = Stddev(self.model)

        X_s = self.ds_raw.air

        alg = GreedyAlgorithm(
            model=self.model,
            X_t=X_s,
            X_s=X_s,
            N_new_context=1,
            task_loader=self.task_loader,
        )

        task = self.task_loader("2014-12-31", context_sampling=10)

        _ = alg(acquisition_fn, task)

    def test_greedy_alg_with_aux_at_targets_without_task_loader_raises_value_error(
        self,
    ):
        acquisition_fn = MeanStddev(self.model)

        X_s = self.ds_raw.air

        alg = GreedyAlgorithm(
            model=self.model_with_aux,
            X_t=X_s,
            X_s=X_s,
            N_new_context=1,
            task_loader=None,  # don't pass task_loader (to raise error)
        )

        task = self.task_loader_with_aux("2014-12-31", context_sampling=10)

        with self.assertRaises(ValueError):
            _ = alg(acquisition_fn, task)

    def test_greedy_alg_with_oracle_acquisition_fn_without_task_loader_raises_value_error(
        self,
    ):
        acquisition_fn = OracleMAE(self.model)

        # Coarsen search points to speed up computation
        X_s = self.ds_raw.air.coarsen(lat=10, lon=10, boundary="trim").mean()

        alg = GreedyAlgorithm(
            model=self.model,
            X_t=X_s,
            X_s=X_s,
            N_new_context=2,
            task_loader=None,  # don't pass task_loader (to raise error)
        )

        task = self.task_loader("2014-12-31", context_sampling=10)

        with self.assertRaises(ValueError):
            _ = alg(acquisition_fn, task)

    def assert_acquisition_fn_without_min_or_max_raises_error(
        self,
    ):
        class DummyAcquisitionFn(AcquisitionFunction):
            """Dummy acquisition function that doesn't set min or max"""

            def __call__(self, **kwargs):
                return np.zeros(1)

        acquisition_fn = DummyAcquisitionFn(self.model)

        X_s = self.ds_raw.air

        with self.assertRaises(ValueError):
            alg = GreedyAlgorithm(
                model=self.model,
                X_t=X_s,
                X_s=X_s,
                N_new_context=2,
            )
