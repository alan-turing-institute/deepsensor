import copy
import os.path
import json
from typing import Union, List, Literal, Optional
import warnings

import lab as B
import numpy as np
import warnings
from matrix import Diagonal
from plum import ModuleType, dispatch

from deepsensor import backend
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task
from deepsensor.model.defaults import (
    compute_greatest_data_density,
    gen_encoder_scales,
    gen_decoder_scale,
)
from deepsensor.model.model import DeepSensorModel
from deepsensor.model.nps import (
    construct_neural_process,
    convert_task_to_nps_args,
    run_nps_model,
    run_nps_model_ar,
)

from neuralprocesses.dist import AbstractMultiOutputDistribution


TFModel = ModuleType("tensorflow.keras", "Model")
TorchModel = ModuleType("torch.nn", "Module")


class ConvNP(DeepSensorModel):
    """A ConvNP regression probabilistic model.

    Wraps around the ``neuralprocesses`` package to construct a ConvNP model.
    See: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/architectures/convgnp.py

    Multiple dispatch is implemented using ``plum`` to allow for re-using the
    model's forward prediction object when computing the logpdf, entropy, etc.
    Alternatively, the model can be run forwards with a ``Task`` object of data
    from the ``TaskLoader``.

    The ``ConvNP`` can optionally be instantiated with:
        - a ``DataProcessor`` object to auto-unnormalise the data at inference time with the ``.predict`` method.
        - a ``TaskLoader`` object to infer sensible default model parameters from the data.

    Many of the ``ConvNP`` class methods utilise multiple dispatch so that they
    can either be run with a ``Task`` object or a ``neuralprocesses`` distribution
    object. This allows for re-using the model's forward prediction object.

    Dimension shapes are expressed in method docstrings in terms of:
        - ``N_features``: number of features/dimensions in the target set.
        - ``N_targets``: number of target points (1D for off-grid targets, 2D for gridded targets).
        - ``N_components``: number of mixture components in the likelihood (for mixture likelihoods only).
        - ``N_samples``: number of samples drawn from the distribution.

    If the model has multiple target sets and the ``Task`` object
    has different target locations for each set, a list of arrays is returned
    for each target set. Otherwise, a single array is returned.

    Examples:
        Instantiate a ``ConvNP`` with all hyperparameters set to their default values:
            >>> ConvNP(data_processor, task_loader)
        Instantiate a ``ConvNP`` and override some hyperparameters:
            >>> ConvNP(data_processor, task_loader, internal_density=250, unet_channels=(128,) * 6)
        Instantiate a ``ConvNP`` with a pre-trained model saved in the folder ``my_trained_model``:
            >>> ConvNP(data_processor, task_loader, model_ID="my_trained_model")
        Instantiate a ``ConvNP`` with an existing ``neuralprocesses`` model object:
            >>> ConvNP(data_processor, task_loader, neural_process=my_neural_process_model)

    Args:
        data_processor (:class:`~.data.processor.DataProcessor`, optional):
            Used for unnormalising model predictions in
            ``.predict`` method.
        task_loader (:class:`~.data.loader.TaskLoader`, optional):
            Used for inferring sensible defaults for hyperparameters
            that are not set by the user.
        model_ID (str, optional):
            Folder to load the model config and weights from. This argument can only
            be used alongside the ``data_processor`` and ``task_loader`` arguments.
        neural_process (TFModel | TorchModel, optional):
            Pre-defined neural process PyTorch/TensorFlow model object. This argument can
            only be used alongside the ``data_processor`` and ``task_loader`` arguments.
        internal_density (int, optional):
            Density of the ConvNP's internal grid (in terms of number of points
            per 1x1 unit square). Defaults to 100.
        likelihood (str, optional):
            Likelihood. Must be one of ``"cnp"`` (equivalently ``"het"``),
            ``"gnp"`` (equivalently ``"lowrank"``), or ``"cnp-spikes-beta"``
            (equivalently ``"spikes-beta"``). Defaults to ``"cnp"``.
        dim_x (int, optional):
            Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional):
            Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional):
            Dimensionality of the outputs of the context set. You should set this
            if the dimensionality of the outputs of the context set is not equal
            to the dimensionality of the outputs of the target set. You should
            also set this if you want to use multiple context sets. In that case,
            set this equal to a tuple of integers indicating the respective output
            dimensionalities.
        dim_yt (int, optional):
            Dimensionality of the outputs of the target set. You should set this
            if the dimensionality of the outputs of the target set is not equal to
            the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional):
            Dimensionality of target-specific auxiliary variables.
        conv_arch (str, optional):
            Convolutional architecture to use. Must be one of
            ``"unet[-res][-sep]"`` or ``"conv[-res][-sep]"``. Defaults to
            ``"unet"``.
        unet_channels (tuple[int], optional):
            Number of channels in the downsampling path of the UNet (including the bottleneck).
            Defaults to four downsampling layers, each with 64 channels. I.e. (64, 64, 64, 64).
            Note: The downsampling path is followed by an upsampling path with the same number of
            channels in the reverse order (plus extra channels for the skip connections).
        unet_kernels (int or tuple[int], optional):
            Sizes of the kernels in the UNet. Defaults to 5.
        unet_resize_convs (bool, optional):
            Use resize convolutions rather than transposed convolutions in the
            UNet. Defaults to ``False``.
        unet_resize_conv_interp_method (str, optional):
            Interpolation method for the resize convolutions in the UNet. Can be
            set to ``"bilinear"``. Defaults to "bilinear".
        num_basis_functions (int, optional):
            Number of basis functions for the low-rank likelihood. Defaults to
            64.
        dim_lv (int, optional):
            Dimensionality of the latent variable. Setting to >0 constructs a
            latent neural process. Defaults to 0.
        encoder_scales (float or tuple[float], optional):
            Initial value for the length scales of the set convolutions for the
            context sets embeddings. Set to a tuple equal to the number of context
            sets to use different values for each set. Set to a single value to use
            the same value for all context sets. Defaults to
            ``1 / internal_density``.
        encoder_scales_learnable (bool, optional):
            Whether the encoder SetConv length scale(s) are learnable. Defaults to
            ``False``.
        decoder_scale (float, optional):
            Initial value for the length scale of the set convolution in the
            decoder. Defaults to ``1 / internal_density``.
        decoder_scale_learnable (bool, optional):
            Whether the decoder SetConv length scale(s) are learnable. Defaults to
            ``False``.
        aux_t_mlp_layers (tuple[int], optional):
            Widths of the layers of the MLP for the target-specific auxiliary
            variable. Defaults to three layers of width 128.
        epsilon (float, optional):
            Epsilon added by the set convolutions before dividing by the density
            channel. Defaults to ``1e-2``.
        dtype (dtype, optional):
            Data type.
    """

    @dispatch
    def __init__(self, *args, **kwargs):
        """Generate a new model using ``construct_neural_process`` with default or
        specified parameters.

        This method does not take a ``TaskLoader`` or ``DataProcessor`` object,
        so the model will not auto-unnormalise predictions at inference time.
        """
        super().__init__()

        self.model, self.config = construct_neural_process(*args, **kwargs)

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        *args,
        verbose: bool = True,
        **kwargs,
    ):
        """Instantiate model from TaskLoader, using data to infer model parameters
        (unless overridden).

        Args:
            data_processor (:class:`~.data.processor.DataProcessor`):
                DataProcessor object. Used for unnormalising model predictions in
                ``.predict`` method.
            task_loader (:class:`~.data.loader.TaskLoader`):
                TaskLoader object. Used for inferring sensible defaults for hyperparameters
                that are not set by the user.
            verbose (bool, optional):
                Whether to print inferred model parameters, by default True.
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
        if "internal_density" not in kwargs:
            internal_density = compute_greatest_data_density(task_loader)
            if verbose:
                print(f"internal_density inferred from TaskLoader: {internal_density}")
            kwargs["internal_density"] = internal_density
        if "encoder_scales" not in kwargs:
            encoder_scales = gen_encoder_scales(kwargs["internal_density"], task_loader)
            if verbose:
                print(f"encoder_scales inferred from TaskLoader: {encoder_scales}")
            kwargs["encoder_scales"] = encoder_scales
        if "decoder_scale" not in kwargs:
            decoder_scale = gen_decoder_scale(kwargs["internal_density"])
            if verbose:
                print(f"decoder_scale inferred from TaskLoader: {decoder_scale}")
            kwargs["decoder_scale"] = decoder_scale

        self.model, self.config = construct_neural_process(*args, **kwargs)
        self._set_num_mixture_components()

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        neural_process: Union[TFModel, TorchModel],
    ):
        """Instantiate with a pre-defined neural process model.

        Args:
            data_processor (:class:`~.data.processor.DataProcessor`):
                DataProcessor object. Used for unnormalising model predictions in
                ``.predict`` method.
            task_loader (:class:`~.data.loader.TaskLoader`):
                TaskLoader object. Used for inferring sensible defaults for hyperparameters
                that are not set by the user.
            neural_process (TFModel | TorchModel):
                Pre-defined neural process PyTorch/TensorFlow model object.
        """
        super().__init__(data_processor, task_loader)

        self.model = neural_process
        self.config = None

    @dispatch
    def __init__(self, model_ID: str):
        """Instantiate a model from a folder containing model weights and config."""
        super().__init__()

        self.load(model_ID)
        self._set_num_mixture_components()

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        model_ID: str,
    ):
        """Instantiate a model from a folder containing model weights and config.

        Args:
            data_processor (:class:`~.data.processor.DataProcessor`):
                dataprocessor object. used for unnormalising model predictions in
                ``.predict`` method.
            task_loader (:class:`~.data.loader.TaskLoader`):
                taskloader object. used for inferring sensible defaults for hyperparameters
                that are not set by the user.
            model_ID (str):
                folder to load the model config and weights from.
        """
        super().__init__(data_processor, task_loader)

        self.load(model_ID)
        self._set_num_mixture_components()

    def _set_num_mixture_components(self):
        """Set the number of mixture components for the model based on the likelihood."""
        if self.config["likelihood"] in ["spikes-beta"]:
            self.N_mixture_components = 3
        elif self.config["likelihood"] in ["bernoulli-gamma"]:
            self.N_mixture_components = 2
        else:
            self.N_mixture_components = 1

    def save(self, model_ID: str):
        """Save the model weights and config to a folder.

        Args:
            model_ID (str):
                Folder to save the model to.

        Returns:
            None.
        """
        os.makedirs(model_ID, exist_ok=True)

        if backend.str == "torch":
            import torch

            torch.save(self.model.state_dict(), os.path.join(model_ID, "model.pt"))
        elif backend.str == "tf":
            self.model.save_weights(os.path.join(model_ID, "model"))
        else:
            raise NotImplementedError(f"Backend {backend.str} not supported.")

        config_fpath = os.path.join(model_ID, "model_config.json")
        with open(config_fpath, "w") as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

    def load(self, model_ID: str):
        """Load a model from a folder containing model weights and config.

        Args:
            model_ID (str):
                Folder to load the model from.

        Returns:
            None.
        """
        config_fpath = os.path.join(model_ID, "model_config.json")
        with open(config_fpath, "r") as f:
            self.config = json.load(f)

        self.model, _ = construct_neural_process(**self.config)

        if backend.str == "torch":
            import torch

            self.model.load_state_dict(torch.load(os.path.join(model_ID, "model.pt")))
        elif backend.str == "tf":
            self.model.load_weights(os.path.join(model_ID, "model"))
        else:
            raise NotImplementedError(f"Backend {backend.str} not supported.")

    def __str__(self):
        return (
            f"ConvNP with config:"
            + "\n"
            + json.dumps(self.config, indent=4, sort_keys=False)
        )

    @classmethod
    def modify_task(cls, task: Task):
        """Cast numpy arrays to TensorFlow or PyTorch tensors, add batch dim, and
        mask NaNs.

        Args:
            task (:class:`~.data.task.Task`):
                ...

        Returns:
            ...: ...
        """
        if "batch_dim" not in task["ops"]:
            task = task.add_batch_dim()
        if "float32" not in task["ops"]:
            task = task.cast_to_float32()
        if "numpy_mask" not in task["ops"]:
            task = task.mask_nans_numpy()
        if "nps_mask" not in task["ops"]:
            task = task.mask_nans_nps()
        if "tensor" not in task["ops"]:
            task = task.convert_to_tensor()

        return task

    def __call__(self, task, n_samples=10, requires_grad=False):
        """Compute ConvNP distribution.

        Args:
            task (:class:`~.data.task.Task`):
                ...
            n_samples (int, optional):
                Number of samples to draw from the distribution, by default 10.
            requires_grad (bool, optional):
                Whether to compute gradients, by default False.

        Returns:
            ...: The ConvNP distribution.
        """
        task = ConvNP.modify_task(task)
        dist = run_nps_model(self.model, task, n_samples, requires_grad)
        return dist

    def _cast_numpy_and_squeeze(
        self,
        x: Union[B.Numeric, List[B.Numeric]],
        squeeze_axes: List[int] = (0, 1),
    ):
        """TODO docstring."""
        if isinstance(x, backend.nps.Aggregate):
            return [np.squeeze(B.to_numpy(xi), axis=squeeze_axes) for xi in x]
        else:
            return np.squeeze(B.to_numpy(x), axis=squeeze_axes)

    def _maybe_concat_multi_targets(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        concat_axis: int = 0,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Concatenate multiple target sets into a single tensor along feature dimension
        and remove size-1 dimensions.

        Args:
            x (:class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]):
                List of target sets.
            squeeze_axes (List[int], optional):
                Axes to squeeze out of the concatenated target sets. Defaults to (0, 1).
            concat_axis (int, optional):
                Axis to concatenate along (*after* squeezing arrays) when
                merging multiple target sets. Defaults to 0.

        Returns:
            (:class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]):
                Concatenated target sets.
        """
        if isinstance(x, (list, tuple)):
            new_list = []
            pos = 0
            for dim in self.task_loader.target_dims:
                new_list.append(x[pos : pos + dim])
                pos += dim
            return [
                B.concat(*[xi for xi in sub_list], axis=concat_axis)
                for sub_list in new_list
            ]
        else:
            return x

    @dispatch
    def mean(self, dist: AbstractMultiOutputDistribution):
        mean = dist.mean
        mean = self._cast_numpy_and_squeeze(mean)
        return self._maybe_concat_multi_targets(mean)

    @dispatch
    def mean(self, task: Task):
        """Mean values of model's distribution at target locations in task.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        Args:
            task (:class:`~.data.task.Task`):
                The task containing the context and target data.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                Mean values.
        """
        dist = self(task)
        return self.mean(dist)

    @dispatch
    def variance(self, dist: AbstractMultiOutputDistribution):
        variance = dist.var
        variance = self._cast_numpy_and_squeeze(variance)
        return self._maybe_concat_multi_targets(variance)

    @dispatch
    def variance(self, task: Task):
        """Variance values of model's distribution at target locations in task.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        Args:
            task (:class:`~.data.task.Task`):
                The task containing the context and target data.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                Variance values.
        """
        dist = self(task)
        return self.variance(dist)

    @dispatch
    def std(self, dist: AbstractMultiOutputDistribution):
        variance = self.variance(dist)
        if isinstance(variance, (list, tuple)):
            return [np.sqrt(v) for v in variance]
        else:
            return np.sqrt(variance)

    @dispatch
    def std(self, task: Task):
        """Standard deviation values of model's distribution at target locations in task.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        Args:
            task (:class:`~.data.task.Task`):
                The task containing the context and target data.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                Standard deviation values.
        """
        dist = self(task)
        return self.std(dist)

    @dispatch
    def alpha(
        self, dist: AbstractMultiOutputDistribution
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if self.config["likelihood"] not in ["spikes-beta", "bernoulli-gamma"]:
            raise NotImplementedError(
                f"ConvNP.alpha method not supported for likelihood {self.config['likelihood']}. "
                f"Try changing the likelihood to a mixture model, e.g. 'spikes-beta' or 'bernoulli-gamma'."
            )
        alpha = dist.slab.alpha
        alpha = self._cast_numpy_and_squeeze(alpha)
        return self._maybe_concat_multi_targets(alpha)

    @dispatch
    def alpha(self, task: Task) -> Union[np.ndarray, List[np.ndarray]]:
        """Alpha parameter values of model's distribution at target locations in task.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        .. note::
            This method only works for models that return a distribution with
            a ``dist.slab.alpha`` attribute, e.g. models with a Beta or
            Bernoulli-Gamma likelihood, where it returns the alpha values of
            the slab component of the mixture model.

        Args:
            task (:class:`~.data.task.Task`):
                The task containing the context and target data.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                Alpha values.
        """
        dist = self(task)
        return self.alpha(dist)

    @dispatch
    def beta(
        self, dist: AbstractMultiOutputDistribution
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if self.config["likelihood"] not in ["spikes-beta", "bernoulli-gamma"]:
            raise NotImplementedError(
                f"ConvNP.beta method not supported for likelihood {self.config['likelihood']}. "
                f"Try changing the likelihood to a mixture model, e.g. 'spikes-beta' or 'bernoulli-gamma'."
            )
        beta = dist.slab.beta
        beta = self._cast_numpy_and_squeeze(beta)
        return self._maybe_concat_multi_targets(beta)

    @dispatch
    def beta(self, task: Task) -> Union[np.ndarray, List[np.ndarray]]:
        """Beta values of model's distribution at target locations in task.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        .. note::
            This method only works for models that return a distribution with
            a ``dist.slab.beta`` attribute, e.g. models with a Beta or
            Bernoulli-Gamma likelihood.

        Args:
            task (:class:`~.data.task.Task`):
                The task containing the context and target data.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                Beta values.
        """
        dist = self(task)
        return self.beta(dist)

    @dispatch
    def mixture_probs(self, dist: AbstractMultiOutputDistribution):
        if self.N_mixture_components == 1:
            raise NotImplementedError(
                f"mixture_probs not supported if model attribute N_mixture_components == 1. "
                f"Try changing the likelihood to a mixture model, e.g. 'spikes-beta'."
            )
        mixture_probs = dist.logprobs
        mixture_probs = self._cast_numpy_and_squeeze(mixture_probs)
        mixture_probs = self._maybe_concat_multi_targets(mixture_probs)
        if isinstance(mixture_probs, (list, tuple)):
            return [np.moveaxis(np.exp(m), -1, 0) for m in mixture_probs]
        else:
            return np.moveaxis(np.exp(mixture_probs), -1, 0)

    @dispatch
    def mixture_probs(self, task: Task):
        """Mixture probabilities of model's distribution at target locations in task.

        Returned numpy arrays have shape ``(N_components, N_features, *N_targets)``.

        Args:
            task (:class:`~.data.task.Task`):
                The task containing the context and target data.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                Mixture probabilities.
        """
        dist = self(task)
        return self.mixture_probs(dist)

    @dispatch
    def covariance(self, dist: AbstractMultiOutputDistribution):
        return B.to_numpy(B.dense(dist.vectorised_normal.var))[0, 0]

    @dispatch
    def covariance(self, task: Task):
        """...

        Args:
            task (:class:`~.data.task.Task`):
                ...

        Returns:
            ...: ...
        """
        dist = self(task)
        return self.covariance(dist)

    @dispatch
    def sample(
        self,
        dist: AbstractMultiOutputDistribution,
        n_samples: int = 1,
    ):
        if self.config["likelihood"] in ["gnp", "lowrank"]:
            samples = dist.noiseless.sample(n_samples)
        else:
            samples = dist.sample(n_samples)
        # Be careful to keep sample dimension in position 0
        samples = self._cast_numpy_and_squeeze(samples, squeeze_axes=(1, 2))
        return self._maybe_concat_multi_targets(samples, concat_axis=1)

    @dispatch
    def sample(self, task: Task, n_samples: int = 1):
        """Create samples from a ConvNP distribution.

        Returned numpy arrays have shape ``(N_samples, N_features, *N_targets)``,

        Args:
            dist (neuralprocesses.dist.AbstractMultiOutputDistribution):
                The distribution to sample from.
            n_samples (int, optional):
                The number of samples to draw from the distribution, by
                default 1.

        Returns:
            :class:`numpy:numpy.ndarray` | List[:class:`numpy:numpy.ndarray`]:
                The samples as an array or list of arrays.
        """
        dist = self(task)
        return self.sample(dist, n_samples)

    @dispatch
    def slice_diag(self, task: Task):
        """Slice out the ConvCNP part of the ConvNP distribution.

        Args:
            task (:class:`~.data.task.Task`):
                The task to slice.

        Returns:
            ...: ...
        """
        dist = self(task)
        if self.config["likelihood"] in ["spikes-beta"]:
            dist_diag = dist
        else:
            dist_diag = backend.nps.MultiOutputNormal(
                dist._mean,
                B.zeros(dist._var),
                Diagonal(B.diag(dist._noise + dist._var)),
                dist.shape,
            )
        return dist_diag

    @dispatch
    def slice_diag(self, dist: AbstractMultiOutputDistribution):
        """Slice out the ConvCNP part of the ConvNP distribution.

        Args:
            dist (neuralprocesses.dist.AbstractMultiOutputDistribution):
                The distribution to slice.

        Returns:
            ...: ...
        """
        if self.config["likelihood"]:
            dist_diag = dist
        else:
            dist_diag = backend.nps.MultiOutputNormal(
                dist._mean,
                B.zeros(dist._var),
                Diagonal(B.diag(dist._noise + dist._var)),
                dist.shape,
            )
        return dist_diag

    @dispatch
    def mean_marginal_entropy(self, dist: AbstractMultiOutputDistribution):
        """Mean marginal entropy over target points given context points.

        Args:
            dist (neuralprocesses.dist.AbstractMultiOutputDistribution):
                The distribution to compute the entropy of.

        Returns:
            float: The mean marginal entropy.
        """
        dist_diag = self.slice_diag(dist)
        return B.mean(B.to_numpy(dist_diag.entropy())[0, 0])

    @dispatch
    def mean_marginal_entropy(self, task: Task):
        """Mean marginal entropy over target points given context points.

        Args:
            task (:class:`~.data.task.Task`):
                The task to compute the entropy of.

        Returns:
            float: The mean marginal entropy.
        """
        dist_diag = self.slice_diag(task)
        return B.mean(B.to_numpy(dist_diag.entropy())[0, 0])

    @dispatch
    def joint_entropy(self, dist: AbstractMultiOutputDistribution):
        """Model entropy over target points given context points.

        Args:
            dist (neuralprocesses.dist.AbstractMultiOutputDistribution):
                The distribution to compute the entropy of.

        Returns:
            float: The model entropy.
        """
        return B.to_numpy(dist.entropy())[0, 0]

    @dispatch
    def joint_entropy(self, task: Task):
        """Model entropy over target points given context points.

        Args:
            task (:class:`~.data.task.Task`):
                The task to compute the entropy of.

        Returns:
            float: The model entropy.
        """
        return B.to_numpy(self(task).entropy())[0, 0]

    @dispatch
    def logpdf(self, dist: AbstractMultiOutputDistribution, task: Task):
        """Joint logpdf over all target sets.

        .. note::
            If the model has multiple target sets, the returned logpdf is the
            mean logpdf over all target sets.

        Args:
            dist (neuralprocesses.dist.AbstractMultiOutputDistribution):
                The distribution to compute the logpdf of.
            task (:class:`~.data.task.Task`):
                The task to compute the logpdf of.

        Returns:
            float: The logpdf.
        """
        # Need to ensure `Y_t` is a tensor and, if multiple target sets,
        #   an nps.Aggregate object
        task = ConvNP.modify_task(task)
        _, _, Y_t, _ = convert_task_to_nps_args(task)
        return B.to_numpy(dist.logpdf(Y_t)).mean()

    @dispatch
    def logpdf(self, task: Task):
        """Joint logpdf over all target sets.

        .. note::
            If the model has multiple target sets, the returned logpdf is the
            mean logpdf over all target sets.

        Args:
            task (:class:`~.data.task.Task`):
                The task to compute the logpdf of.

        Returns:
            float: The logpdf.
        """
        dist = self(task)
        return self.logpdf(dist, task)

    def loss_fn(
        self,
        task: Task,
        fix_noise=None,
        num_lv_samples: int = 8,
        normalise: bool = False,
    ):
        """Compute the loss of a task.

        Args:
            task (:class:`~.data.task.Task`):
                The task to compute the loss of.
            fix_noise (...):
                Whether to fix the noise to the value specified in the model
                config.
            num_lv_samples (int, optional):
                If latent variable model, number of lv samples for evaluating
                the loss, by default 8.
            normalise (bool, optional):
                Whether to normalise the loss by the number of target points,
                by default False.

        Returns:
            float: The loss.
        """
        task = ConvNP.modify_task(task)

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
        n_samples: int = 1,
        X_target_AR: Optional[np.ndarray] = None,
        ar_subsample_factor: int = 1,
        fill_type: Literal["mean", "sample"] = "mean",
    ):
        """Autoregressive sampling from the model.

        AR sampling with optional functionality to only draw AR samples over a
        subset of the target set and then infill the rest of the sample with
        the model mean or joint sample conditioned on the AR samples.

        Returned numpy arrays have shape ``(N_samples, N_features, *N_targets)``,

        .. note::
            AR sampling only works for 0th context/target set, and only for models with
            a single target set.

        Args:
            task (:class:`~.data.task.Task`):
                The task to sample from.
            n_samples (int, optional):
                The number of samples to draw from the distribution, by
                default 1.
            X_target_AR (:class:`numpy:numpy.ndarray`, optional):
                Locations to draw AR samples over. If None, AR samples will be
                drawn over the target locations in the task. Defaults to None.
            ar_subsample_factor (int, optional):
                Subsample target locations to draw AR samples over. Defaults
                to 1.
            fill_type (Literal["mean", "sample"], optional):
                How to infill the rest of the sample. Must be one of "mean" or
                "sample". Defaults to "mean".

        Returns:
            :class:`numpy:numpy.ndarray`
                The samples.
        """
        if len(task["X_t"]) > 1 or (task["Y_t"] is not None and len(task["Y_t"]) > 1):
            raise NotImplementedError(
                "AR sampling with multiple target sets is not supported."
            )

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

        task = task.flatten_gridded_data()
        task_arsample = task_arsample.flatten_gridded_data()

        task_arsample = ConvNP.modify_task(task_arsample)
        task = ConvNP.modify_task(task)

        if backend.str == "torch":
            import torch

            # Run AR sampling with torch.no_grad() to avoid prohibitive backprop computation for AR
            with torch.no_grad():
                (
                    mean,
                    variance,
                    noiseless_samples,
                    noisy_samples,
                ) = run_nps_model_ar(self.model, task_arsample, num_samples=n_samples)
        else:
            (
                mean,
                variance,
                noiseless_samples,
                noisy_samples,
            ) = run_nps_model_ar(self.model, task_arsample, num_samples=n_samples)

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
                    # Should this be a `.sample` call?
                    pred = self.mean(task_with_sample)
                elif fill_type == "sample":
                    # Sample from joint distribution over all target locations
                    pred = self.sample(task_with_sample, n_samples=1)

                full_samples.append(pred)
            full_samples = np.stack(full_samples, axis=0)

            return full_samples
        else:
            return noiseless_samples[:, 0]  # Slice out batch dim


def concat_tasks(tasks: List[Task], multiple: int = 1) -> Task:
    """
    Concatenate a list of tasks into a single task containing multiple batches.

    ```{warning}
    `concat_tasks` has been moved to deepsensor.data.task and will be removed from "
        "deepsensor.model.convnp in a future release.
    ```
    """
    warnings.warn(
        "concat_tasks has been moved to deepsensor.data.task and will be removed from "
        "deepsensor.model.convnp in a future release.",
        FutureWarning,
    )
    return deepsensor.data.task.concat_tasks(tasks, multiple)
