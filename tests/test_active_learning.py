import unittest
import xarray as xr
import numpy as np

import deepsensor.tensorflow as deepsensor
from deepsensor.active_learning.acquisition_fns import (
    AcquisitionFunction,
    MeanVariance,
    MeanStddev,
    pNormStddev,
    MeanMarginalEntropy,
    JointEntropy,
    OracleMAE,
    OracleRMSE,
    OracleMarginalNLL,
    OracleJointNLL,
    Stddev,
    ExpectedImprovement,
    Random,
    ContextDist,
)
from deepsensor.active_learning.algorithms import GreedyAlgorithm

from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor, xarray_to_coord_array_normalised
from deepsensor.model.convnp import ConvNP


# from deepsensor.active_learning.acquisition_fns import


class TestActiveLearning(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # It's safe to share data between tests because the TaskLoader does not modify data
        ds_raw = xr.tutorial.open_dataset("air_temperature")["air"]
        self.ds_raw = ds_raw
        self.data_processor = DataProcessor(x1_name="lat", x2_name="lon")
        self.ds = self.data_processor(ds_raw)
        # Set up a model with two context sets and two target sets for generality
        self.task_loader = TaskLoader(
            context=[self.ds, self.ds], target=[self.ds, self.ds]
        )
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
        for context_set_idx, target_set_idx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            sequential_acquisition_fns = [
                MeanStddev(self.model, context_set_idx, target_set_idx),
                MeanVariance(self.model, context_set_idx, target_set_idx),
                pNormStddev(self.model, context_set_idx, target_set_idx, p=3),
                MeanMarginalEntropy(self.model, context_set_idx, target_set_idx),
                JointEntropy(self.model, context_set_idx, target_set_idx),
                OracleMAE(self.model, context_set_idx, target_set_idx),
                OracleRMSE(self.model, context_set_idx, target_set_idx),
                OracleMarginalNLL(self.model, context_set_idx, target_set_idx),
                OracleJointNLL(self.model, context_set_idx, target_set_idx),
            ]
            parallel_acquisition_fns = [
                Stddev(self.model, context_set_idx, target_set_idx),
                ExpectedImprovement(self.model, context_set_idx, target_set_idx),
                ContextDist(self.model, context_set_idx, target_set_idx),
                Random(self.model, context_set_idx, target_set_idx),
            ]

            # Coarsen search points to speed up computation
            X_s = self.ds_raw.coarsen(lat=10, lon=10, boundary="trim").mean()
            X_s = self.data_processor.map_coords(X_s)
            X_s_arr = xarray_to_coord_array_normalised(X_s)

            task = self.task_loader(
                "2014-12-31", context_sampling=10, target_sampling="all"
            )

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
        X_s = self.ds_raw.coarsen(lat=10, lon=10, boundary="trim").mean()
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
        X_s = self.ds_raw.coarsen(lat=10, lon=10, boundary="trim").mean()
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
        X_s = self.ds_raw.coarsen(lat=10, lon=10, boundary="trim").mean()
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
        X_s = self.ds_raw
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
        X_s = self.ds_raw
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
        X_s = self.ds_raw
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

    def test_acquisition_fn_without_min_or_max_raises_error(
        self,
    ):
        class DummyAcquisitionFn(AcquisitionFunction):
            """Dummy acquisition function that doesn't set min or max"""

            def __call__(self, **kwargs):
                return np.zeros(1)

        acquisition_fn = DummyAcquisitionFn(self.model)

        X_s = self.ds_raw
        alg = GreedyAlgorithm(
            model=self.model,
            X_t=X_s,
            X_s=X_s,
            N_new_context=2,
        )
        with self.assertRaises(ValueError):
            _ = alg(acquisition_fn, None)

    def test_parallel_acquisition_fn_with_diff_raises_error(
        self,
    ):
        acquisition_fn = Stddev(self.model)
        X_s = self.ds_raw
        alg = GreedyAlgorithm(
            model=self.model,
            X_t=X_s,
            X_s=X_s,
        )
        task = self.task_loader(
            "2014-12-31", context_sampling=10, target_sampling="all"
        )

        # This should work
        _ = alg(acquisition_fn, task)
        # This should raise an error
        with self.assertRaises(ValueError):
            _ = alg(acquisition_fn, task, diff=True)
