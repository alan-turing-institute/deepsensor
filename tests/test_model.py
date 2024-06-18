import copy
import itertools
import tempfile

from parameterized import parameterized

import os
import xarray as xr
import numpy as np
import pandas as pd
import unittest

import lab as B

import deepsensor.torch as deepsensor

from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader
from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import Trainer

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
    """A test class for the ``ConvNP`` model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # It's safe to share data between tests because the TaskLoader does not modify data
        self.da = _gen_data_xr()
        self.df = _gen_data_pandas()

        self.dp = DataProcessor()
        _ = self.dp([self.da, self.df])  # Compute normalisation parameters

    def _gen_task_loader_call_args(self, n_context, n_target):
        """Generate arguments for TaskLoader.__call__"""
        for sampling_method in [
            10,
            0.5,
            "all",
        ]:
            yield [sampling_method] * n_context, [sampling_method] * n_target

    @parameterized.expand(range(1, 4))
    def test_model_call(self, n_context_and_target):
        """Check ``ConvNP`` runs with all possible combinations of context/target
        sampling methods.
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

            for (
                context_sampling,
                target_sampling,
            ) in self._gen_task_loader_call_args(
                n_context_and_target, n_context_and_target
            ):
                task = tl("2020-01-01", context_sampling, target_sampling)
                dist = model(task)

        return None

    @parameterized.expand(range(1, 4))
    def test_prediction_shapes_lowlevel(self, n_target_sets):
        """Test low-level model prediction interface over a range of number of
        target sets.
        """
        # Make dataset 5D for non-trivial target dimensions
        ndim = 5
        ds = xr.Dataset(
            {f"var{i}": self.da for i in range(ndim)},
        )
        tl = TaskLoader(
            context=ds,
            target=[ds] * n_target_sets,
        )

        context_sampling = 10

        likelihoods = ["cnp", "gnp", "cnp-spikes-beta"]

        dim_y_combined = sum(tl.target_dims)

        for likelihood in likelihoods:
            model = ConvNP(
                self.dp,
                tl,
                unet_channels=(5, 5, 5),
                likelihood=likelihood,
                verbose=False,
            )
            assert dim_y_combined == model.config["dim_yt"]

            for target_sampling, expected_obs_shape in (
                # expected obs shape is (10,) when target_sampling is 10
                (10, (10,)),
                # expected obs shape is da.shape[-2:] when target_sampling is "all"
                ("all", self.da.shape[-2:]),
            ):
                task = tl("2020-01-01", context_sampling, target_sampling)

                n_targets = np.product(expected_obs_shape)

                # Tensors
                mean = model.mean(task)
                # TODO avoid repeated code by looping over methods
                if isinstance(mean, (list, tuple)):
                    for m, dim_y in zip(mean, tl.target_dims):
                        assert_shape(m, (dim_y, *expected_obs_shape))
                else:
                    assert_shape(mean, (dim_y_combined, *expected_obs_shape))

                variance = model.variance(task)
                if isinstance(variance, (list, tuple)):
                    for v, dim_y in zip(variance, tl.target_dims):
                        assert_shape(v, (dim_y, *expected_obs_shape))
                else:
                    assert_shape(variance, (dim_y_combined, *expected_obs_shape))

                stddev = model.stddev(task)
                if isinstance(stddev, (list, tuple)):
                    for s, dim_y in zip(stddev, tl.target_dims):
                        assert_shape(s, (dim_y, *expected_obs_shape))
                else:
                    assert_shape(stddev, (dim_y_combined, *expected_obs_shape))

                n_samples = 5
                samples = model.sample(task, n_samples)
                if isinstance(samples, (list, tuple)):
                    for s, dim_y in zip(samples, tl.target_dims):
                        assert_shape(s, (n_samples, dim_y, *expected_obs_shape))
                else:
                    assert_shape(
                        samples, (n_samples, dim_y_combined, *expected_obs_shape)
                    )

                if likelihood in ["cnp", "gnp"]:
                    n_target_dims = np.product(tl.target_dims)
                    assert_shape(
                        model.covariance(task),
                        (
                            n_targets * dim_y_combined * n_target_dims,
                            n_targets * dim_y_combined * n_target_dims,
                        ),
                    )
                if likelihood in ["cnp-spikes-beta"]:
                    mixture_probs = model.mixture_probs(task)
                    if isinstance(mixture_probs, (list, tuple)):
                        for p, dim_y in zip(mixture_probs, tl.target_dims):
                            assert_shape(
                                p,
                                (
                                    model.N_mixture_components,
                                    dim_y,
                                    *expected_obs_shape,
                                ),
                            )
                    else:
                        assert_shape(
                            mixture_probs,
                            (
                                model.N_mixture_components,
                                dim_y_combined,
                                *expected_obs_shape,
                            ),
                        )

                    x = model.alpha(task)
                    if isinstance(x, (list, tuple)):
                        for p, dim_y in zip(x, tl.target_dims):
                            assert_shape(p, (dim_y, *expected_obs_shape))
                    else:
                        assert_shape(x, (dim_y_combined, *expected_obs_shape))

                    x = model.beta(task)
                    if isinstance(x, (list, tuple)):
                        for p, dim_y in zip(x, tl.target_dims):
                            assert_shape(p, (dim_y, *expected_obs_shape))
                    else:
                        assert_shape(x, (dim_y_combined, *expected_obs_shape))

                # Scalars
                if likelihood in ["cnp", "gnp"]:
                    # Methods for Gaussian likelihoods only
                    x = model.joint_entropy(task)
                    assert x.size == 1 and x.shape == ()
                    x = model.mean_marginal_entropy(task)
                    assert x.size == 1 and x.shape == ()
                x = model.logpdf(task)
                assert x.size == 1 and x.shape == ()
                x = B.to_numpy(model.loss_fn(task))
                assert x.size == 1 and x.shape == ()

    @parameterized.expand(range(1, 4))
    def test_prediction_shapes_highlevel(self, target_dim):
        """Test high-level ``.predict`` interface over a range of number of target"""
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
        pred = model.predict(
            tasks,
            X_t=self.da,
            n_samples=n_samples,
            unnormalise=(
                True if target_dim == 1 else False
            ),  # TODO fix unnormalising for multiple equally named targets
        )
        assert [isinstance(ds, xr.Dataset) for ds in pred.values()]
        for var_ID in pred:
            assert_shape(
                pred[var_ID]["mean"],
                (len(dates), self.da.x1.size, self.da.x2.size),
            )
            assert_shape(
                pred[var_ID]["std"],
                (len(dates), self.da.x1.size, self.da.x2.size),
            )
            for sample_i in range(n_samples):
                assert_shape(
                    pred[var_ID][f"sample_{sample_i}"],
                    (len(dates), self.da.x1.size, self.da.x2.size),
                )

        # Offgrid predictions: test pandas `X_t` and numpy `X_t`
        n_samples = 5
        for X_t in [self.df.loc[dates[0]], np.zeros((2, 4))]:
            pred = model.predict(
                tasks,
                X_t=X_t,
                n_samples=n_samples,
                unnormalise=False if target_dim > 1 else True,
            )
            assert [isinstance(df, pd.DataFrame) for df in pred.values()]
            if isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index)):
                N_t = len(X_t)
            elif isinstance(X_t, np.ndarray):
                N_t = X_t.shape[-1]
            n_preds = len(dates) * N_t

            for var_ID in pred:
                assert_shape(pred[var_ID]["mean"], (n_preds,))
                assert_shape(pred[var_ID]["std"], (n_preds,))
                for sample_i in range(n_samples):
                    assert_shape(pred[var_ID][f"sample_{sample_i}"], (n_preds,))

    @parameterized.expand(range(1, 4))
    def test_nans_offgrid_context(self, ndim):
        """Test that ``ConvNP`` can handle ``NaN``s in offgrid context."""
        tl = TaskLoader(
            context=_gen_data_xr(data_vars=range(ndim)),
            target=self.da,
        )

        # All NaNs
        task = tl("2020-01-01", context_sampling=10, target_sampling=10)
        task["Y_c"][0][:, 0] = np.nan
        task["Y_t"][0][:, 0] = np.nan  # Also sanity check with target
        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)
        _ = model(task)

        # One NaN
        task = tl("2020-01-01", context_sampling=10, target_sampling=10)
        task["Y_c"][0][0, 0] = np.nan
        task["Y_t"][0][0, 0] = np.nan  # Also sanity check with target
        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)
        _ = model(task)

    @parameterized.expand(range(1, 4))
    def test_nans_gridded_context(self, ndim):
        """Test that ``ConvNP`` can handle ``NaN``s in gridded context."""
        tl = TaskLoader(
            context=_gen_data_xr(data_vars=range(ndim)),
            target=self.da,
        )

        # All NaNs
        task = tl("2020-01-01", context_sampling="all", target_sampling="all")
        task["Y_c"][0][:, 0, 0] = np.nan
        task["Y_t"][0][:, 0, 0] = np.nan  # Also sanity check with target
        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)
        _ = model(task)

        # One NaN
        task = tl("2020-01-01", context_sampling="all", target_sampling="all")
        task["Y_c"][0][0, 0, 0] = np.nan
        task["Y_t"][0][0, 0, 0] = np.nan  # Also sanity check with target
        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)
        _ = model(task)

    def test_nans_in_context(self):
        """Test nothing breaks when NaNs present in context."""
        tl = TaskLoader(context=self.da, target=self.da)
        task = tl("2020-01-01", context_sampling=10, target_sampling=10)

        # Convert first observation of context to NaN
        task["Y_c"][0][0] = np.nan

        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)

        # Check that nothing breaks
        model(task)

    def test_highlevel_predict_coords_align_with_X_t_ongrid(self):
        """Test coordinates of the xarray returned predictions align with the
        coordinates of X_t.
        """
        # Instantiate an xarray object that would lead to rounding errors
        region_size = (61, 81)
        lat_lims = (30, 75)
        lon_lims = (-15, 45)
        latitude = np.linspace(*lat_lims, region_size[0], dtype=np.float32)
        longitude = np.linspace(*lon_lims, region_size[1], dtype=np.float32)
        dummy_data = np.random.normal(size=(1, *region_size))
        da_raw = xr.DataArray(
            dummy_data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "latitude": latitude,
                "longitude": longitude,
            },
            name="dummy_data",
        )

        dp = DataProcessor(x1_name="latitude", x2_name="longitude")
        da = dp(da_raw)

        tl = TaskLoader(context=da, target=da)
        model = ConvNP(dp, tl, unet_channels=(5, 5, 5), verbose=False)
        task = tl("2020-01-01")
        pred = model.predict(task, X_t=da_raw)

        assert np.array_equal(
            pred["dummy_data"]["mean"]["latitude"], da_raw["latitude"]
        )
        assert np.array_equal(
            pred["dummy_data"]["mean"]["longitude"], da_raw["longitude"]
        )

    def test_highlevel_predict_coords_align_with_X_t_offgrid(self):
        """Test coordinates of the pandas returned predictions align with the
        coordinates of X_t.
        """
        # Instantiate a pandas object that would lead to rounding errors
        region_size = (61, 81)
        lat_lims = (30, 75)
        lon_lims = (-15, 45)
        latitude = np.linspace(*lat_lims, region_size[0], dtype=np.float32)
        longitude = np.linspace(*lon_lims, region_size[1], dtype=np.float32)
        dummy_data = np.random.normal(size=(region_size)).ravel()
        df_raw = pd.DataFrame(
            dummy_data,
            index=pd.MultiIndex.from_product(
                [[pd.Timestamp("2020-01-01")], latitude, longitude],
                names=["time", "latitude", "longitude"],
            ),
            columns=["dummy_data"],
        )

        dp = DataProcessor(
            x1_name="latitude",
            x2_name="longitude",
        )
        df = dp(df_raw)

        tl = TaskLoader(context=df, target=df)
        model = ConvNP(dp, tl, unet_channels=(5, 5, 5), verbose=False)
        task = tl("2020-01-01")
        pred = model.predict(task, X_t=df_raw.loc["2020-01-01"])

        assert np.array_equal(
            pred["dummy_data"]["mean"].reset_index()["latitude"],
            df_raw.reset_index()["latitude"],
        )
        assert np.array_equal(
            pred["dummy_data"]["mean"].reset_index()["longitude"],
            df_raw.reset_index()["longitude"],
        )

    def test_highlevel_predict_with_pred_params_pandas(self):
        """Test that passing ``pred_params`` to ``.predict`` works with
        a spikes-beta likelihood for prediction to pandas.
        """
        tl = TaskLoader(context=self.da, target=self.da)
        model = ConvNP(
            self.dp,
            tl,
            unet_channels=(5, 5, 5),
            verbose=False,
            likelihood="cnp-spikes-beta",
        )
        task = tl("2020-01-01", context_sampling=10, target_sampling=10)

        # Off-grid prediction
        X_t = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])

        # Check that nothing breaks and the correct parameters are returned
        pred_params = ["mean", "std", "variance", "alpha", "beta"]
        pred = model.predict(task, X_t=X_t, pred_params=pred_params)
        for pred_param in pred_params:
            assert pred_param in pred["var"]

        # Test mixture probs special case
        pred_params = ["mixture_probs"]
        pred = model.predict(task, X_t=self.da, pred_params=pred_params)
        for component in range(model.N_mixture_components):
            pred_param = f"mixture_probs_{component}"
            assert pred_param in pred["var"]

    def test_highlevel_predict_with_pred_params_xarray(self):
        """Test that passing ``pred_params`` to ``.predict`` works with
        a spikes-beta likelihood for prediction to xarray.
        """
        tl = TaskLoader(context=self.da, target=self.da)
        model = ConvNP(
            self.dp,
            tl,
            unet_channels=(5, 5, 5),
            verbose=False,
            likelihood="cnp-spikes-beta",
        )
        task = tl("2020-01-01", context_sampling=10, target_sampling=10)

        # Check that nothing breaks and the correct parameters are returned
        pred_params = ["mean", "std", "variance", "alpha", "beta"]
        pred = model.predict(task, X_t=self.da, pred_params=pred_params)
        for pred_param in pred_params:
            assert pred_param in pred["var"]

        # Test mixture probs special case
        pred_params = ["mixture_probs"]
        pred = model.predict(task, X_t=self.da, pred_params=pred_params)
        for component in range(model.N_mixture_components):
            pred_param = f"mixture_probs_{component}"
            assert pred_param in pred["var"]

    def test_highlevel_predict_with_invalid_pred_params(self):
        """Test that passing ``pred_params`` to ``.predict`` works."""
        tl = TaskLoader(context=self.da, target=self.da)
        model = ConvNP(
            self.dp,
            tl,
            unet_channels=(5, 5, 5),
            verbose=False,
            likelihood="cnp-spikes-beta",
        )
        task = tl("2020-01-01", context_sampling=10, target_sampling=10)

        # Check that passing an invalid parameter raises an AttributeError
        with self.assertRaises(AttributeError):
            model.predict(task, X_t=self.da, pred_params=["invalid_param"])

    def test_saving_and_loading(self):
        """Test saving and loading of model"""
        with tempfile.TemporaryDirectory() as folder:
            ds_raw = xr.tutorial.open_dataset("air_temperature")

            data_processor = DataProcessor(x1_name="lat", x2_name="lon")
            ds = data_processor(ds_raw)

            t2m_fpath = f"{folder}/air_temperature_normalised.nc"
            ds.to_netcdf(t2m_fpath)

            task_loader = TaskLoader(context=t2m_fpath, target=t2m_fpath)

            model = ConvNP(
                data_processor,
                task_loader,
                unet_channels=(5,) * 3,
                verbose=False,
            )

            # Train the model for a few iterations to test the trained model is restored correctly later.
            task = task_loader("2014-12-31", 40, "all", datewise_deterministic=True)
            trainer = Trainer(model)
            for _ in range(10):
                trainer([task])
            pred_before = model.predict(task, X_t=ds_raw)

            data_processor.save(folder)
            task_loader.save(folder)
            model.save(folder)

            data_processor_loaded = DataProcessor(folder)
            task_loader_loaded = TaskLoader(folder)
            model_loaded = ConvNP(data_processor_loaded, task_loader_loaded, folder)

            task = task_loader_loaded("2014-12-31", 40, datewise_deterministic=True)
            pred_loaded = model_loaded.predict(task, X_t=ds_raw)

            xr.testing.assert_allclose(
                pred_loaded["air"]["mean"], pred_before["air"]["mean"]
            )
            print("Means match")

            xr.testing.assert_allclose(
                pred_loaded["air"]["std"], pred_before["air"]["std"]
            )
            print("Standard deviations match")

    def test_running_convnp_without_X_t_raises_value_error(self):
        """Test that running ConvNP with Task not containing X_t raises a ValueError"""
        tl = TaskLoader(context=self.da, target=self.da)
        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)
        # Task with `None` for X_t entry
        task = tl("2020-01-01", context_sampling=10, target_sampling=None)
        with self.assertRaises(ValueError):
            _ = model(task)

    def test_ar_sample(self):
        """Test autoregressive sampling in the high-level ``.predict`` interface."""
        tl = TaskLoader(context=self.da, target=self.da)
        model = ConvNP(self.dp, tl, unet_channels=(5, 5, 5), verbose=False)
        dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        task = tl(dates, context_sampling=10)
        n_samples = 5

        pred = model.predict(
            task,
            X_t=self.da,
            n_samples=n_samples,
            ar_sample=True,
            ar_subsample_factor=4,
        )
        for var_ID in pred:
            for sample_i in range(n_samples):
                assert_shape(
                    pred[var_ID][f"sample_{sample_i}"],
                    (len(dates), self.da.x1.size, self.da.x2.size),
                )

        with self.assertRaises(ValueError):
            # Not passing `n_samples` along with `ar_sample=True` should raise a ValueError
            pred = model.predict(
                task,
                X_t=self.da,
                ar_sample=True,
            )


def assert_shape(x, shape: tuple):
    """Assert that the shape of ``x`` matches ``shape``."""
    # TODO put this in a utils module?
    assert len(x.shape) == len(shape), (x.shape, shape)
    for _a, _b in zip(x.shape, shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, shape)
