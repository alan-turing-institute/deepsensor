from deepsensor import backend
from deepsensor.model.defaults import gen_encoder_scales, gen_ppu
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task
from deepsensor.model.nps import (
    convert_task_to_nps_args,
    run_nps_model,
    construct_neural_process,
)

import copy

from typing import List, Union

import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import lab as B
from matrix import Diagonal
from plum import dispatch

# For dispatching with TF and PyTorch model types when they have not yet been loaded.
# See https://beartype.github.io/plum/types.html#moduletype
from plum import ModuleType

TFModel = ModuleType("tensorflow.keras", "Model")
TorchModel = ModuleType("torch.nn", "Module")


def create_empty_prediction_array(
    dates,
    resolution_factor,
    X_t,
    coord_names={"x1": "x1", "x2": "x2"},
    data_vars=["mean", "std"],
):
    x1_lowres = X_t.coords[coord_names["x1"]]
    x2_lowres = X_t.coords[coord_names["x2"]]

    x1_hires = np.linspace(
        x1_lowres[0],
        x1_lowres[-1],
        int(x1_lowres.size * resolution_factor),
        dtype="float32",
    )
    x2_hires = np.linspace(
        x2_lowres[0],
        x2_lowres[-1],
        int(x2_lowres.size * resolution_factor),
        dtype="float32",
    )

    dims = ["time", coord_names["x1"], coord_names["x2"]]
    coords = {
        "time": pd.to_datetime(dates),
        coord_names["x1"]: x1_hires,
        coord_names["x2"]: x2_hires,
    }

    pred_ds = xr.Dataset(
        {data_var: xr.DataArray(dims=dims, coords=coords) for data_var in data_vars},
    ).astype("float32")

    # Convert time coord to pandas timestamps
    pred_ds = pred_ds.assign_coords(time=pd.to_datetime(pred_ds.time.values))

    # TODO: Convert init time to forecast time?
    # pred_ds = pred_ds.assign_coords(
    #     time=pred_ds['time'] + pd.Timedelta(days=task_loader.target_delta_t[0]))

    return pred_ds


class ProbabilisticModel:

    """
    Base class for probabilistic model used for DeepSensor.
    Ensures a set of methods required for DeepSensor
    are implemented by specific model classes that inherit from it.
    """

    def __init__(self):
        pass

    def mean(self, dataset, *args, **kwargs):
        """
        Computes the model mean prediction over target points based on given context
        data.
        """
        raise NotImplementedError()

    def covariance(self, dataset, *args, **kwargs):
        """
        Computes the model covariance matrix over target points based on given context
        data. Shape (N, N).
        """
        raise NotImplementedError()

    def variance(self, dataset, *args, **kwargs):
        """
        Model marginal variance over target points given context points.
        Shape (N,).
        """
        raise NotImplementedError()

    def stddev(self, dataset):
        """
        Model marginal standard deviation over target points given context points.
        Shape (N,).
        """
        var = self.variance(dataset)
        return var**0.5

    def entropy(self, dataset, *args, **kwargs):
        """
        Computes the model entropy over target points based on given context
        data.
        """
        raise NotImplementedError()

    def logpdf(self, dataset, *args, **kwargs):
        """
        Computes the joint model logpdf over target points based on given context
        data.
        """
        raise NotImplementedError()

    def loss(self, dataset, *args, **kwargs):
        """
        Computes the model loss over target points based on given context data.
        """
        raise NotImplementedError()

    def sample(self, dataset, n_samples=1, *args, **kwargs):
        """
        Draws `n_samples` joint samples over target points based on given context
        data.
        returned shape is (n_samples, n_target).
        """
        raise NotImplementedError()

    def mutual_information(
        self, dataset, X_new, Y_new, *args, context_set_idx=0, **kwargs
    ):
        """
        WIP: Old code not using new dataset data structure.

        Computes the mutual information over target set T given context set C and
        the new (proposed) context set N:
            I(T|C;N) = H(T|C) - H(T|C,N)

        Uses the subclassed `entropy` method.
        """

        dataset_with_new = concat_obs_to_dataset(dataset, X_new, Y_new, context_set_idx)

        entropy_before = self.entropy(dataset)

        entropy_after = self.entropy(dataset_with_new)

        return entropy_before - entropy_after


class DeepSensorModel(ProbabilisticModel):

    """
    Implements DeepSensor prediction functionality of a ProbabilisticModel.
    Allows for outputting an xarray object containing on-grid predictions or a pandas
    object containing off-grid predictions.
    """

    def __init__(
        self,
        data_processor: DataProcessor = None,
        task_loader: TaskLoader = None,
    ):
        """Initialise DeepSensorModel

        :param task_loader: TaskLoader object, used to determine target variables for unnormalising
        :param data_processor: DataProcessor object, used to unnormalise predictions
        """
        self.task_loader = task_loader
        self.data_processor = data_processor

    def predict(
        self,
        tasks: Union[List[Task], Task],
        X_t: Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, pd.Index],
        X_t_normalised: bool = False,
        resolution_factor=1,
        n_samples=0,
        noiseless_samples=True,
        seed=0,
        progress_bar=0,
        verbose=False,
    ):
        """Predict on a regular grid or at off-grid locations.

        TODO:
        - Test with multiple targets model

        :param tasks: List of tasks containing context data.
        :param X_t: Target locations to predict at. Can be an xarray object containing
            on-grid locations or a pandas object containing off-grid locations.
        :param X_t_normalised: Whether the `X_t` coords are normalised.
            If False, will normalise the coords before passing to model. Default False.
        :param resolution_factor: Optional factor to increase the resolution of the
            target grid by. E.g. 2 will double the target resolution, 0.5 will halve it.
            Applies to on-grid predictions only. Default 1.
        :param n_samples: Number of joint samples to draw from the model.
            If 0, will not draw samples. Default 0.
        :param noiseless_samples: Whether to draw noiseless samples from the model. Default True.
        :param seed: Random seed for deterministic sampling. Default 0.
        :param progress_bar: Whether to display a progress bar over tasks. Default 0.
        :param verbose: Whether to print time taken for prediction. Default False.

        Returns:
            - If X_t is a pandas object, returns pandas objects containing off-grid predictions.
            - If X_t is an xarray object, returns xarray object containing on-grid predictions.
            - If n_samples == 0, returns only mean and std predictions.
            - If n_samples > 0, returns mean, std and samples predictions.
        """
        tic = time.time()

        if type(tasks) is Task:
            tasks = [tasks]

        if n_samples >= 1:
            B.set_random_seed(seed)
            np.random.seed(seed)

        dates = [task["time"] for task in tasks]

        target_var_IDs = self.task_loader.target_var_IDs[
            0
        ]  # TEMP just first target set

        if isinstance(X_t, pd.Index):
            X_t = pd.DataFrame(index=X_t)

        if not X_t_normalised:
            X_t = self.data_processor.map_coords(X_t)

        if isinstance(X_t, (xr.DataArray, xr.Dataset)):
            mode = "on-grid"
        elif isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index)):
            mode = "off-grid"

        if mode == "on-grid":
            mean = create_empty_prediction_array(
                dates, resolution_factor, X_t, data_vars=target_var_IDs
            ).to_array(dim="data_var")
            std = create_empty_prediction_array(
                dates, resolution_factor, X_t, data_vars=target_var_IDs
            ).to_array(dim="data_var")
            if n_samples >= 1:
                samples = create_empty_prediction_array(
                    dates, resolution_factor, X_t, data_vars=target_var_IDs
                ).to_array(dim="data_var")
                samples = samples.expand_dims(
                    dim=dict(sample=np.arange(n_samples))
                ).copy()

            X_t_arr = (mean["x1"].values, mean["x2"].values)

        elif mode == "off-grid":
            # Repeat target locs for each date to create multiindex
            idxs = [(date, *idxs) for date in dates for idxs in X_t.index]
            index = pd.MultiIndex.from_tuples(idxs, names=["time", *X_t.index.names])
            mean = pd.DataFrame(index=index, columns=target_var_IDs)
            std = pd.DataFrame(index=index, columns=target_var_IDs)
            if n_samples >= 1:
                idxs_samples = [
                    (sample, date, *idxs)
                    for sample in range(n_samples)
                    for date in dates
                    for idxs in X_t.index
                ]
                index_samples = pd.MultiIndex.from_tuples(
                    idxs_samples,
                    names=["sample", "time", *X_t.index.names],
                )
                samples = pd.DataFrame(index=index_samples, columns=target_var_IDs)

            X_t_arr = X_t.reset_index()[["x1", "x2"]].values.T

        for task in tqdm(tasks, position=0, disable=progress_bar < 1, leave=True):
            # TODO - repeat based on number of targets?
            task["X_t"] = [X_t_arr]

            # Run model forwards once to generate output distribution
            dist = self(task, n_samples=n_samples)

            if mode == "on-grid":
                mean.loc[:, task["time"], :, :] = self.mean(dist)
                std.loc[:, task["time"], :, :] = self.stddev(dist)
                if n_samples >= 1:
                    B.set_random_seed(seed)
                    np.random.seed(seed)
                    samples.loc[:, :, task["time"], :, :] = self.sample(
                        dist, n_samples=n_samples, noiseless=noiseless_samples
                    )
            elif mode == "off-grid":
                # TODO multi-target case
                mean.loc[task["time"]] = self.mean(dist).T
                std.loc[task["time"]] = self.stddev(dist).T
                if n_samples >= 1:
                    B.set_random_seed(seed)
                    np.random.seed(seed)
                    samples_arr = self.sample(
                        dist, n_samples=n_samples, noiseless=noiseless_samples
                    )
                    for sample_i in range(n_samples):
                        samples.loc[sample_i, task["time"]] = samples_arr[sample_i].T

        if mode == "on-grid":
            mean = mean.to_dataset(dim="data_var")
            std = std.to_dataset(dim="data_var")
            if n_samples >= 1:
                samples = samples.to_dataset(dim="data_var")

        if self.task_loader is not None and self.data_processor is not None:
            mean = self.data_processor.unnormalise(mean)
            std = self.data_processor.unnormalise(std, add_offset=False)
            if n_samples >= 1:
                samples = self.data_processor.unnormalise(samples)

        if verbose:
            dur = time.time() - tic
            print(f"Done in {np.floor(dur / 60)}m:{dur % 60:.0f}s.\n")

        if n_samples >= 1:
            return mean, std, samples
        else:
            return mean, std


class ConvNP(DeepSensorModel):

    """
    Implements a ConvNP regression probabilistic model.

    Multiple dispatch is implemented using `plum` to allow for re-using
    the model's forward prediction object when computing the logpdf, entropy, etc.
    Alternatively, the model can be run forwards with a `task` dictionary of data
    from the `DataLoader`.

    """

    @dispatch
    def __init__(self, *args, **kwargs):
        """Generate a new model using `nps.construct_convgnp` with default or specified parameters

        This method does not take a `TaskLoader` or `DataProcessor` object, so the model
        will not auto-unnormalise predictions at inference time.
        """
        # The parent class will instantiate with `task_loader` and `data_processor` set to None,
        # so unnormalisation will not be performed at inference time.
        super().__init__()

        self.model = construct_neural_process(
            *args,
            **kwargs,
        )

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        *args,
        verbose: bool = True,
        **kwargs,
    ):
        """Instantiate model from TaskLoader, using data to infer model parameters (unless overridden)

        Args:
            data_processor (DataProcessor, optional): DataProcessor object. Defaults to None.
            task_loader (TaskLoader): TaskLoader object
        """
        super().__init__(data_processor, task_loader)

        if "dim_yc" not in kwargs:
            dim_yc = task_loader.context_dims
            if verbose:
                print(f"dim_yc inferred from TaskLoader: {dim_yc}")
            kwargs["dim_yc"] = dim_yc
        if "dim_yt" not in kwargs:
            dim_yt = sum(task_loader.target_dims)  # Must be an int
            if verbose:
                print(f"dim_yt inferred from TaskLoader: {dim_yt}")
            kwargs["dim_yt"] = dim_yt
        if "points_per_unit" not in kwargs:
            ppu = gen_ppu(task_loader)
            if verbose:
                print(f"points_per_unit inferred from TaskLoader: {ppu}")
            kwargs["points_per_unit"] = ppu
        if "encoder_scales" not in kwargs:
            encoder_scales = gen_encoder_scales(kwargs["points_per_unit"], task_loader)
            if verbose:
                print(f"encoder_scales inferred from TaskLoader: {encoder_scales}")
            kwargs["encoder_scales"] = encoder_scales

        self.model = construct_neural_process(
            dim_x=2,
            *args,
            **kwargs,
        )

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        neural_process: Union[TFModel, TorchModel],
    ):
        """Instantiate with a pre-defined neural process model"""
        super().__init__(data_processor, task_loader)

        self.model = neural_process

    @classmethod
    def modify_task(cls, task):
        """Cast numpy arrays to TensorFlow or PyTorch tensors, add batch dim, and mask NaNs"""

        def array_modify_fn(arr):
            arr = arr[np.newaxis, ...]  # Add batch dim

            arr = arr.astype(np.float32)  # Cast to float32

            # Find NaNs and keep size-1 variable dim
            mask = np.any(np.isnan(arr), axis=1, keepdims=False)
            if np.any(mask):
                # Set NaNs to zero - necessary for `neural_process`
                arr[mask] = 0.0

            # Convert to tensor object based on deep learning backend
            arr = backend.convert_to_tensor(arr)

            # Convert to `nps.Masked` object if there are NaNs
            if B.any(mask):
                arr = backend.nps.Masked(arr, B.cast(B.dtype(arr), mask))

            return arr

        task = task.modify(array_modify_fn, modify_flag="NPS")
        return task

    @classmethod
    def check_task(cls, task):
        """Check that the task is compatible with the model."""
        if task["flag"] is None:
            task = cls.modify_task(task)
        elif task["flag"] != "NPS":
            raise ValueError(f"Task has been modified for {task['modify']}.")
        return task

    def __call__(self, task, n_samples=10, requires_grad=False):
        """Compute ConvNP distribution."""
        task = ConvNP.check_task(task)
        dist = run_nps_model(self.model, task, n_samples, requires_grad)
        return dist

    @dispatch
    def mean(self, dist: backend.nps.AbstractMultiOutputDistribution):
        return B.to_numpy(dist.mean)[0, 0]

    @dispatch
    def mean(self, task: Task):
        return B.to_numpy(self(task).mean)[0, 0]

    @dispatch
    def entropy(self, dist: backend.nps.AbstractMultiOutputDistribution):
        """Model entropy over target points given context points."""
        return B.to_numpy(dist.entropy())[0]

    @dispatch
    def entropy(self, task: Task):
        return B.to_numpy(self(task).entropy())[0]

    @dispatch
    def covariance(self, dist: backend.nps.AbstractMultiOutputDistribution):
        return B.to_numpy(B.dense(dist.vectorised_normal.var))[0]

    @dispatch
    def covariance(self, task: Task):
        return B.to_numpy(B.dense(self(task).vectorised_normal.var))[0]

    @dispatch
    def variance(self, dist: backend.nps.AbstractMultiOutputDistribution):
        return B.to_numpy(dist.var)[0, 0]

    @dispatch
    def variance(self, task: Task):
        return B.to_numpy(self(task).var)[0, 0]

    @dispatch
    def logpdf(
        self,
        dist: backend.nps.AbstractMultiOutputDistribution,
        task: Task,
        station_set_idx=0,
    ):
        # Need Y_target to be the right shape for model in the event that task is from the
        # default DataLoader... is this the best way to do this?
        task = ConvNP.check_task(task)

        Y_target = task["Y_t"][station_set_idx]
        return B.to_numpy(dist.logpdf(Y_target)).mean()

    @dispatch
    def logpdf(self, task: Task, station_set_idx=0):
        # Need Y_target to be the right shape for model in the event that task is from the
        # default DataLoader... is this the best way to do this?
        task = ConvNP.check_task(task)

        Y_target = task["Y_t"][station_set_idx]
        return B.to_numpy(self(task).logpdf(Y_target)).mean()

    def loss_fn(self, task, fix_noise=None, num_lv_samples=8, normalise=False):
        """

        Parameters
        ----------
        model_config
        neural_process
        task
        num_lv_samples: If latent variable model, number of lv samples for evaluating the loss
        normalise

        Returns
        -------

        """
        task = ConvNP.check_task(task)
        context_data, xt, yt, model_kwargs = convert_task_to_nps_args(task)

        logpdfs = backend.nps.loglik(
            self.model,
            context_data,
            xt,
            yt,
            **model_kwargs,
            fix_noise=fix_noise,
            num_samples=num_lv_samples,
            normalise=normalise,
        )

        loss = -B.mean(logpdfs)

        return loss

    @dispatch
    def sample(
        self,
        dist: backend.nps.AbstractMultiOutputDistribution,
        n_samples=1,
        noiseless=True,
    ):
        if noiseless:
            return B.to_numpy(dist.noiseless.sample(n_samples))[:, 0, 0]  # first batch
        else:
            return B.to_numpy(dist.sample(n_samples))[:, 0, 0]

    @dispatch
    def sample(self, task: Task, n_samples=1, noiseless=True):
        if noiseless:
            return B.to_numpy(self(task).noiseless.sample(n_samples))[
                :, 0, 0
            ]  # first batch
        else:
            return B.to_numpy(self(task).sample(n_samples))[:, 0, 0]

    @dispatch
    def slice_diag(self, task: Task):
        """Slice out the ConvCNP part of the ConvNP distribution."""
        dist = self(task)
        dist_diag = backend.nps.MultiOutputNormal(
            dist._mean,
            B.zeros(dist._var),
            Diagonal(B.diag(dist._noise + dist._var)),
            dist.shape,
        )
        return dist_diag

    @dispatch
    def slice_diag(self, dist: backend.nps.AbstractMultiOutputDistribution):
        """Slice out the ConvCNP part of the ConvNP distribution."""
        dist_diag = backend.nps.MultiOutputNormal(
            dist._mean,
            B.zeros(dist._var),
            Diagonal(B.diag(dist._noise + dist._var)),
            dist.shape,
        )
        return dist_diag

    def ar_sample(
        self, task: Task, n_samples=1, X_target_AR=None, ar_subsample_factor=1
    ):
        """AR sampling with optional functionality to only draw AR samples over a subset of the
        target set and then infull the rest of the sample with the model mean conditioned on the
        AR samples."""
        if X_target_AR is not None:
            # User has specified a set of locations to draw AR samples over
            task_arsample = copy.deepcopy(task)
            task_arsample["X_t"][0] = X_target_AR
        elif ar_subsample_factor > 1:
            # Subsample target locations to draw AR samples over
            task_arsample = copy.deepcopy(task)
            xt = task["X_t"][0]
            ndim_2d = np.sqrt(xt.shape[-1])
            if int(ndim_2d) == ndim_2d:
                # Targets on a square: subsample targets for AR along both spatial dimension
                xt_2d = xt.reshape(-1, int(ndim_2d), int(ndim_2d))
                xt = xt_2d[..., ::ar_subsample_factor, ::ar_subsample_factor].reshape(
                    2, -1
                )
            else:
                xt = xt[..., ::ar_subsample_factor]
            task_arsample["X_t"][0] = xt
        else:
            task_arsample = copy.deepcopy(task)

        task_arsample = ConvNP.check_task(task_arsample)
        task = ConvNP.check_task(task)

        mean, variance, noiseless_samples, noisy_samples = run_nps_model_ar(
            self.model, task_arsample, num_samples=n_samples
        )

        # Slice out first (and assumed only) target entry in nps.Aggregate object
        noiseless_samples = B.to_numpy(noiseless_samples)[0]

        if ar_subsample_factor > 1 or X_target_AR is not None:
            # AR sample locations not equal to target locations - infill the rest of the
            # sample with the model mean conditioned on the AR samples
            full_samples = []
            for sample in noiseless_samples:
                task_with_sample = copy.deepcopy(task)
                task_with_sample["X_c"][0] = np.concatenate(
                    [task["X_c"][0], task_arsample["X_t"][0]], axis=-1
                )
                task_with_sample["Y_c"][0] = np.concatenate(
                    [task["Y_c"][0], sample], axis=-1
                )

                if type(self) is ConvCNP:
                    # Independent predictions for each target location - just compute the mean
                    # conditioned on the AR samples
                    pred = self.mean(
                        task_with_sample
                    )  # Should this be a `.sample` call?
                else:
                    # Sample from joint distribution over all target locations
                    # NOTE Risky to assume all model classes other than `ConvCNP` model correlations.
                    pred = self.sample(task_with_sample, n_samples=1)

                full_samples.append(pred)
            full_samples = np.stack(full_samples, axis=0)

            return full_samples
        else:
            return noiseless_samples[:, 0]  # Slice out batch dim
