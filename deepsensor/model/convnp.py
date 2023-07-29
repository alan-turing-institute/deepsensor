import copy
import warnings
from typing import Union, List

import lab as B
import numpy as np
from matrix import Diagonal
from plum import ModuleType, dispatch

from deepsensor import backend
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import (
    Task,
    flatten_gridded_data_in_task,
    flatten_X,
    flatten_Y,
)
from deepsensor.model.defaults import gen_ppu, gen_encoder_scales, gen_decoder_scale
from deepsensor.model.model import DeepSensorModel
from deepsensor.model.nps import (
    construct_neural_process,
    convert_task_to_nps_args,
    run_nps_model,
    run_nps_model_ar,
)

TFModel = ModuleType("tensorflow.keras", "Model")
TorchModel = ModuleType("torch.nn", "Module")


class ConvNP(DeepSensorModel):

    """A ConvNP regression probabilistic model.

    Wraps around the `neuralprocesses` package to construct a ConvNP model.
    See: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/architectures/convgnp.py

    Multiple dispatch is implemented using `plum` to allow for re-using
    the model's forward prediction object when computing the logpdf, entropy, etc.
    Alternatively, the model can be run forwards with a `Task` object of data
    from the `TaskLoader`.

    The `ConvNP` can optionally be instantiated with:
    - a `DataProcessor` object to auto-unnormalise the data at inference time with the `.predict` method.
    - a `TaskLoader` object to infer sensible default model parameters from the data.

    These additional parameters can be passed to the `__init__` method to
    customise the model, which will override any defaults inferred from a `TaskLoader`.
        points_per_unit (int, optional): Density of the internal discretisation.
            Defaults to 100.
        likelihood (str, optional): Likelihood. Must be one of `"cnp"` (equivalently `"het"`),
            `"gnp"` (equivalently `"lowrank"`), or `"cnp-spikes-beta"` (equivalently `"spikes-beta"`).
            Defaults to `"cnp"`.
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional): Dimensionality of target-specific auxiliary
            variables.
        conv_arch (str, optional): Convolutional architecture to use. Must be one of
            `"unet[-res][-sep]"` or `"conv[-res][-sep]"`. Defaults to `"unet"`.
        unet_channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        unet_kernels (int or tuple[int], optional): Sizes of the kernels in the UNet.
            Defaults to 5.
        unet_resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions in the UNet. Defaults to `False`.
        unet_resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions in the UNet. Can be set to `"bilinear"`. Defaults
            to "bilinear".
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to `64`.
        dim_lv (int, optional): Dimensionality of the latent variable. Setting to >0
             constructs a latent neural process. Defaults to 0.
        encoder_scales (float or tuple[float], optional): Initial value for the length
            scales of the set convolutions for the context sets embeddings. Set to a tuple
            equal to the number of context sets to use different values for each set.
            Set to a single value to use the same value for all context sets.
            Defaults to `1 / points_per_unit`.
        encoder_scales_learnable (bool, optional): Whether the encoder SetConv
            length scale(s) are learnable. Defaults to `False`.
        decoder_scale (float, optional): Initial value for the length scale of the
            set convolution in the decoder. Defaults to `1 / points_per_unit`.
        decoder_scale_learnable (bool, optional): Whether the decoder SetConv
            length scale(s) are learnable. Defaults to `False`.
        aux_t_mlp_layers (tuple[int], optional): Widths of the layers of the MLP
            for the target-specific auxiliary variable. Defaults to three layers of
            width 128.
        epsilon (float, optional): Epsilon added by the set convolutions before
            dividing by the density channel. Defaults to `1e-2`.
        dtype (dtype, optional): Data type.
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

        self.model = construct_neural_process(*args, **kwargs)

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
        if "dim_aux_t" not in kwargs:
            dim_aux_t = task_loader.aux_at_target_dims
            if verbose:
                print(f"dim_aux_t inferred from TaskLoader: {dim_aux_t}")
            kwargs["dim_aux_t"] = dim_aux_t
        if "aux_t_mlp_layers" not in kwargs and kwargs["dim_aux_t"] > 0:
            kwargs["aux_t_mlp_layers"] = (64,) * 3
            if verbose:
                print(f"Setting aux_t_mlp_layers: {kwargs['aux_t_mlp_layers']}")
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
        if "decoder_scale" not in kwargs:
            decoder_scale = gen_decoder_scale(kwargs["points_per_unit"])
            if verbose:
                print(f"decoder_scale inferred from TaskLoader: {decoder_scale}")
            kwargs["decoder_scale"] = decoder_scale

        self.model = construct_neural_process(*args, dim_x=2, **kwargs)

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

            # Find NaNs
            mask = np.isnan(arr)
            if np.any(mask):
                # Set NaNs to zero - necessary for `neuralprocesses` (can't have NaNs)
                arr[mask] = 0.0
                # Mask array (True for observed, False for missing) - keep size 1 variable dim
                mask = ~np.any(mask, axis=1, keepdims=True)

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
        mean = dist.mean
        if isinstance(mean, backend.nps.Aggregate):
            return [B.to_numpy(m)[0, 0] for m in mean]
        else:
            return B.to_numpy(mean)[0, 0]

    @dispatch
    def mean(self, task: Task):
        dist = self(task)
        return self.mean(dist)

    @dispatch
    def variance(self, dist: backend.nps.AbstractMultiOutputDistribution):
        variance = dist.var
        if isinstance(variance, backend.nps.Aggregate):
            return [B.to_numpy(v)[0, 0] for v in variance]
        else:
            return B.to_numpy(variance)[0, 0]

    @dispatch
    def variance(self, task: Task):
        dist = self(task)
        return self.variance(dist)

    @dispatch
    def stddev(self, dist: backend.nps.AbstractMultiOutputDistribution):
        variance = self.variance(dist)
        if isinstance(variance, (list, tuple)):
            return [np.sqrt(v) for v in variance]
        else:
            return np.sqrt(variance)

    @dispatch
    def stddev(self, task: Task):
        dist = self(task)
        return self.stddev(dist)

    @dispatch
    def covariance(self, dist: backend.nps.AbstractMultiOutputDistribution):
        return B.to_numpy(B.dense(dist.vectorised_normal.var))[0, 0]

    @dispatch
    def covariance(self, task: Task):
        dist = self(task)
        return self.covariance(dist)

    @dispatch
    def sample(
        self,
        dist: backend.nps.AbstractMultiOutputDistribution,
        n_samples=1,
        noiseless=True,
    ):
        if noiseless:
            samples = dist.noiseless.sample(n_samples)
        else:
            samples = dist.sample(n_samples)

        if isinstance(samples, backend.nps.Aggregate):
            return [B.to_numpy(s)[:, 0, 0] for s in samples]
        else:
            return B.to_numpy(samples)[:, 0, 0]

    @dispatch
    def sample(self, task: Task, n_samples=1, noiseless=True):
        dist = self(task)
        return self.sample(dist, n_samples, noiseless)

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

    @dispatch
    def mean_marginal_entropy(self, dist: backend.nps.AbstractMultiOutputDistribution):
        """Mean marginal entropy over target points given context points."""
        dist_diag = self.slice_diag(dist)
        return B.mean(B.to_numpy(dist_diag.entropy())[0, 0])

    @dispatch
    def mean_marginal_entropy(self, task: Task):
        """Mean marginal entropy over target points given context points."""
        dist_diag = self.slice_diag(task)
        return B.mean(B.to_numpy(dist_diag.entropy())[0, 0])

    @dispatch
    def joint_entropy(self, dist: backend.nps.AbstractMultiOutputDistribution):
        """Model entropy over target points given context points."""
        return B.to_numpy(dist.entropy())[0, 0]

    @dispatch
    def joint_entropy(self, task: Task):
        return B.to_numpy(self(task).entropy())[0, 0]

    @dispatch
    def logpdf(self, dist: backend.nps.AbstractMultiOutputDistribution, task: Task):
        # Model outputs joint distribution over all targets: Concat targets along observation dimension
        Y_t = B.concat(*task["Y_t"], axis=-1)
        return B.to_numpy(dist.logpdf(Y_t)).mean()

    @dispatch
    def logpdf(self, task: Task):
        dist = self(task)
        return self.logpdf(dist, task)

    def loss_fn(self, task: Task, fix_noise=None, num_lv_samples=8, normalise=False):
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
        # Remove NaNs from the target data if present
        task, nans_present = remove_nans_from_task_Y_t_if_present(task)
        # if nans_present:
        #     TODO raise error like:
        #     warnings.warn(
        #         "NaNs present in the target data. These will be removed before evaluating the loss.",
        #
        #     )

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

    def ar_sample(
        self,
        task: Task,
        n_samples=1,
        X_target_AR=None,
        ar_subsample_factor=1,
        fill_type="mean",
    ):
        """Autoregressive sampling from the model.

        AR sampling with optional functionality to only draw AR samples over a subset of the
        target set and then infill the rest of the sample with the model mean or joint sample
        conditioned on the AR samples.

        NOTE:
            - AR sampling only works for 0th context/target set
        """

        # AR sampling requires gridded data to be flattened, not coordinate tuples
        task_arsample = copy.deepcopy(task)
        task = copy.deepcopy(task)

        if X_target_AR is not None:
            # User has specified a set of locations to draw AR samples over
            task_arsample["X_t"][0] = X_target_AR
        elif ar_subsample_factor > 1:
            # Subsample target locations to draw AR samples over
            xt = task["X_t"][0]
            if isinstance(xt, tuple):
                # Targets on a grid: subsample targets for AR along spatial dimension
                xt = (
                    xt[0][..., ::ar_subsample_factor],
                    xt[1][..., ::ar_subsample_factor],
                )
            else:
                xt = xt[..., ::ar_subsample_factor]
            task_arsample["X_t"][0] = xt
        else:
            task_arsample = copy.deepcopy(task)

        task = flatten_gridded_data_in_task(task)
        task_arsample = flatten_gridded_data_in_task(task_arsample)

        task_arsample = ConvNP.check_task(task_arsample)
        task = ConvNP.check_task(task)

        if backend.str == "torch":
            import torch

            # Run AR sampling with torch.no_grad() to avoid prohibitive backprop computation for AR
            with torch.no_grad():
                mean, variance, noiseless_samples, noisy_samples = run_nps_model_ar(
                    self.model, task_arsample, num_samples=n_samples
                )
        else:
            mean, variance, noiseless_samples, noisy_samples = run_nps_model_ar(
                self.model, task_arsample, num_samples=n_samples
            )

        # Slice out first (and assumed only) target entry in nps.Aggregate object
        noiseless_samples = B.to_numpy(noiseless_samples)

        if ar_subsample_factor > 1 or X_target_AR is not None:
            # AR sample locations not equal to target locations - infill the rest of the
            # sample with the model mean conditioned on the AR samples
            full_samples = []
            for sample in noiseless_samples:
                task_with_sample = copy.deepcopy(task)
                task_with_sample["X_c"][0] = B.concat(
                    task["X_c"][0], task_arsample["X_t"][0], axis=-1
                )
                task_with_sample["Y_c"][0] = B.concat(task["Y_c"][0], sample, axis=-1)

                if fill_type == "mean":
                    # Compute the mean conditioned on the AR samples
                    pred = self.mean(
                        task_with_sample
                    )  # Should this be a `.sample` call?
                elif fill_type == "sample":
                    # Sample from joint distribution over all target locations
                    pred = self.sample(task_with_sample, n_samples=1)

                full_samples.append(pred)
            full_samples = np.stack(full_samples, axis=0)

            return full_samples
        else:
            return noiseless_samples[:, 0]  # Slice out batch dim


def concat_tasks(tasks: List[Task], multiple: int = 1) -> Task:
    """Concatenate a list of tasks into a single task containing multiple batches.

    TODO:
    - Consider moving to `nps.py` as this leverages `neuralprocesses` functionality.
    - Raise error if aux_t values passed (not supported I don't think)

    Parameters
    ----------
    tasks : list of Task. List of tasks to concatenate into a single task.
    multiple : int. Contexts are padded to the smallest multiple of this number that is greater
        than the number of contexts in each task. Defaults to 1 (padded to the largest number of
        contexts in the tasks). Setting to a larger number will increase the amount of padding
        but decrease the range of tensor shapes presented to the model, which simplifies
        the computational graph in graph mode.

    Returns
    -------
    merged_task : Task. Task containing multiple batches.
    """
    if len(tasks) == 1:
        return tasks[0]

    # Assert number of target sets equal
    n_target_sets = [len(task["Y_t"]) for task in tasks]
    if not all([n == n_target_sets[0] for n in n_target_sets]):
        raise ValueError(
            f"All tasks must have the same number of target sets to concatenate: got {n_target_sets}. "
        )
    n_target_sets = n_target_sets[0]

    for target_set_i in range(n_target_sets):
        # Raise error if target sets have different numbers of targets across tasks
        n_target_obs = [task["Y_t"][target_set_i].size for task in tasks]
        if not all([n == n_target_obs[0] for n in n_target_obs]):
            raise ValueError(
                f"All tasks must have the same number of targets to concatenate: got {n_target_sets}. "
                "If you want to train using batches containing tasks with differing numbers of targets, "
                "you can run the model individually over each task and average the losses."
            )

        # Raise error if target sets are different types (gridded/non-gridded) across tasks
        if isinstance(tasks[0]["X_t"][target_set_i], tuple):
            for task in tasks:
                if not isinstance(task["X_t"][target_set_i], tuple):
                    raise ValueError(
                        "All tasks must have the same type of target set (gridded or non-gridded) "
                        f"to concatenate. For target set {target_set_i}, got {type(task['X_t'][target_set_i])}."
                    )

    # For each task, store list of tuples of (x_c, y_c) (one tuple per context set)
    contexts = []
    for i, task in enumerate(tasks):
        # Ensure converted to tensors with batch dims
        task = ConvNP.modify_task(task)
        tasks[i] = task

        contexts_i = list(zip(task["X_c"], task["Y_c"]))
        contexts.append(contexts_i)

    # List of tuples of merged (x_c, y_c) along batch dim with padding
    # (up to the smallest multiple of `multiple` greater than the number of contexts in each task)
    merged_context = [
        backend.nps.merge_contexts(
            *[context_set for context_set in contexts_i], multiple=multiple
        )
        for contexts_i in zip(*contexts)
    ]

    merged_task = copy.deepcopy(tasks[0])

    # Convert list of tuples of (x_c, y_c) to list of x_c and list of y_c
    merged_task["X_c"] = [c[0] for c in merged_context]
    merged_task["Y_c"] = [c[1] for c in merged_context]

    # This assumes that all tasks have the same number of targets
    for i in range(n_target_sets):
        if isinstance(tasks[0]["X_t"][i], tuple):
            # Target set is gridded with tuple of coords for `X_t`
            merged_task["X_t"][i] = (
                B.concat(*[t["X_t"][i][0] for t in tasks], axis=0),
                B.concat(*[t["X_t"][i][1] for t in tasks], axis=0),
            )
        else:
            # Target set is off-the-grid with tensor for `X_t`
            merged_task["X_t"][i] = B.concat(*[t["X_t"][i] for t in tasks], axis=0)
        merged_task["Y_t"][i] = B.concat(*[t["Y_t"][i] for t in tasks], axis=0)

    merged_task["time"] = [t["time"] for t in tasks]

    merged_task = Task(merged_task)

    return merged_task


def remove_nans_from_task_Y_t_if_present(task):
    """If NaNs are present in task["Y_t"], remove them (and corresponding task["X_t"])"""
    Y_t_nans_list = []
    # First check whether there are any NaNs that we need to remove
    nans_present = False
    for Y_t_nans in Y_t_nans_list:
        if B.any(Y_t_nans):
            nans_present = True

    if nans_present:
        for i, (X, Y) in enumerate(zip(task["X_t"], task["Y_t"])):
            Y = flatten_Y(Y)
            Y_t_nans = B.any(B.isnan(Y), axis=0)  # shape (n_targets,)
            Y_t_nans_list.append(Y_t_nans)

    if not nans_present:
        return task, False

    # NaNs present in task - make deep copy and remove NaNs
    task = copy.deepcopy(task)
    for i, (X, Y, Y_t_nans) in enumerate(zip(task["X_t"], task["Y_t"], Y_t_nans_list)):
        if B.any(Y_t_nans):
            if isinstance(X, tuple):
                # Gridded data
                X = flatten_X(X)
                Y = flatten_Y(Y)
            task["X_t"][i] = X[:, ~Y_t_nans]
            task["Y_t"][i] = Y[:, ~Y_t_nans]
            if "Y_t_aux" in task.keys():
                task["Y_t_aux"] = task["Y_t_aux"][:, ~Y_t_nans]

    return task, True
