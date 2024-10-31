from .. import backend
import lab as B

from deepsensor.data.task import Task
from typing import Tuple, Optional, Literal


def convert_task_to_nps_args(task: Task):
    """Infer & build model call signature from ``task`` dict.

    ..
        TODO move to ConvNP class?

    Args:
        task (:class:`~.data.task.Task`):
            Task object containing context and target sets.

    Returns:
        tuple[list[tuple[numpy.ndarray, numpy.ndarray]], numpy.ndarray, numpy.ndarray, dict]:
            ...
    """
    context_data = list(zip(task["X_c"], task["Y_c"]))

    if task["X_t"] is None:
        raise ValueError(
            f"Running `neuralprocesses` model with no target locations (got {task['X_t']}). "
            f"Have you not provided a `target_sampling` argument to `TaskLoader`?"
        )
    elif len(task["X_t"]) == 1 and task["Y_t"] is None:
        xt = task["X_t"][0]
        yt = None
    elif len(task["X_t"]) > 1 and task["Y_t"] is None:
        # Multiple target sets, different target locations
        xt = backend.nps.AggregateInput(*[(xt, i) for i, xt in enumerate(task["X_t"])])
        yt = None
    elif len(task["X_t"]) == 1 and len(task["Y_t"]) == 1:
        # Single target set
        xt = task["X_t"][0]
        yt = task["Y_t"][0]
    elif len(task["X_t"]) > 1 and len(task["Y_t"]) > 1:
        # Multiple target sets, different target locations
        assert len(task["X_t"]) == len(task["Y_t"])
        xts = []
        yts = []
        target_dims = [yt.shape[1] for yt in task["Y_t"]]
        # Map from ND target sets to 1D target sets
        dim_counter = 0
        for i, (xt, yt) in enumerate(zip(task["X_t"], task["Y_t"])):
            # Repeat target locations for each target dimension in target set
            xts.extend([(xt, dim_counter + j) for j in range(target_dims[i])])
            yts.extend([yt[:, j : j + 1] for j in range(target_dims[i])])
            dim_counter += target_dims[i]
        xt = backend.nps.AggregateInput(*xts)
        yt = backend.nps.Aggregate(*yts)
    elif len(task["X_t"]) == 1 and len(task["Y_t"]) > 1:
        # Multiple target sets, same target locations; `Y_t`s along feature dim
        xt = task["X_t"][0]
        yt = B.concat(*task["Y_t"], axis=1)
    else:
        raise ValueError(
            f"Incorrect target locations and target observations (got {len(task['X_t'])} and {len(task['Y_t'])})"
        )

    model_kwargs = {}
    if "Y_t_aux" in task.keys():
        model_kwargs["aux_t"] = task["Y_t_aux"]

    return context_data, xt, yt, model_kwargs


def run_nps_model(
    neural_process,
    task: Task,
    n_samples: Optional[int] = None,
    requires_grad: bool = False,
):
    """Run ``neuralprocesses`` model.

    Args:
        neural_process (neuralprocesses.Model):
            Neural process model.
        task (:class:`~.data.task.Task`):
            Task object containing context and target sets.
        n_samples (int, optional):
            Number of samples to draw from the model. Defaults to ``None``
            (single sample).
        requires_grad (bool, optional):
            Whether to require gradients. Defaults to ``False``.

    Returns:
        neuralprocesses.distributions.Distribution:
            Distribution object containing the model's predictions.
    """
    context_data, xt, _, model_kwargs = convert_task_to_nps_args(task)
    if backend.str == "torch" and not requires_grad:
        # turn off grad
        import torch

        with torch.no_grad():
            dist = neural_process(
                context_data, xt, **model_kwargs, num_samples=n_samples
            )
    else:
        dist = neural_process(context_data, xt, **model_kwargs, num_samples=n_samples)
    return dist


def run_nps_model_ar(neural_process, task: Task, num_samples: int = 1):
    """Run ``neural_process`` in AR mode.

    Args:
        neural_process (neuralprocesses.Model):
            Neural process model.
        task (:class:`~.data.task.Task`):
            Task object containing context and target sets.
        num_samples (int, optional):
            Number of samples to draw from the model. Defaults to 1.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            Tuple of mean, variance, noiseless samples, and noisy samples.
    """
    context_data, xt, _, _ = convert_task_to_nps_args(task)

    # NOTE can't use `model_kwargs` in AR mode (ie can't use auxiliary MLP at targets)
    mean, variance, noiseless_samples, noisy_samples = backend.nps.ar_predict(
        neural_process,
        context_data,
        xt,
        num_samples=num_samples,
    )

    return mean, variance, noiseless_samples, noisy_samples


def construct_neural_process(
    dim_x: int = 2,
    dim_yc: int = 1,
    dim_yt: int = 1,
    dim_aux_t: Optional[int] = None,
    dim_lv: int = 0,
    conv_arch: str = "unet",
    unet_channels: Tuple[int, ...] = (64, 64, 64, 64),
    unet_resize_convs: bool = True,
    unet_resize_conv_interp_method: Literal["bilinear"] = "bilinear",
    aux_t_mlp_layers: Optional[Tuple[int, ...]] = None,
    likelihood: Literal["cnp", "gnp", "cnp-spikes-beta"] = "cnp",
    unet_kernels: int = 5,
    internal_density: int = 100,
    encoder_scales: float = 1 / 100,
    encoder_scales_learnable: bool = False,
    decoder_scale: float = 1 / 100,
    decoder_scale_learnable: bool = False,
    num_basis_functions: int = 64,
    epsilon: float = 1e-2,
):
    """Construct a ``neuralprocesses`` ConvNP model.

    See: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/architectures/convgnp.py

    Docstring below modified from ``neuralprocesses``. If more kwargs are
    needed, they must be explicitly passed to ``neuralprocesses`` constructor
    (not currently safe to use `**kwargs` here).

    Args:
        dim_x (int, optional):
            Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional):
            Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional):
            Dimensionality of the outputs of the context set. You should set
            this if the dimensionality of the outputs of the context set is not
            equal to the dimensionality of the outputs of the target set. You
            should also set this if you want to use multiple context sets. In
            that case, set this equal to a tuple of integers indicating the
            respective output dimensionalities.
        dim_yt (int, optional):
            Dimensionality of the outputs of the target set. You should set
            this if the dimensionality of the outputs of the target set is not
            equal to the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional):
            Dimensionality of target-specific auxiliary variables.
        internal_density (int, optional):
            Density of the ConvNP's internal grid (in terms of number of points
            per 1x1 unit square). Defaults to 100.
        likelihood (str, optional):
            Likelihood. Must be one of ``"cnp"`` (equivalently ``"het"``),
            ``"gnp"`` (equivalently ``"lowrank"``), or ``"cnp-spikes-beta"``
            (equivalently ``"spikes-beta"``). Defaults to ``"cnp"``.
        conv_arch (str, optional):
            Convolutional architecture to use. Must be one of
            ``"unet[-res][-sep]"`` or ``"conv[-res][-sep]"``. Defaults to
            ``"unet"``.
        unet_channels (tuple[int], optional):
            Channels of every layer of the UNet. Defaults to six layers each
            with 64 channels.
        unet_kernels (int or tuple[int], optional):
            Sizes of the kernels in the UNet. Defaults to 5.
        unet_resize_convs (bool, optional):
            Use resize convolutions rather than transposed convolutions in the
            UNet. Defaults to ``False``.
        unet_resize_conv_interp_method (str, optional):
            Interpolation method for the resize convolutions in the UNet. Can
            be set to ``"bilinear"``. Defaults to "bilinear".
        num_basis_functions (int, optional):
            Number of basis functions for the low-rank likelihood. Defaults to
            64.
        dim_lv (int, optional):
            Dimensionality of the latent variable. Setting to >0 constructs a
            latent neural process. Defaults to 0.
        encoder_scales (float or tuple[float], optional):
            Initial value for the length scales of the set convolutions for the
            context sets embeddings. Set to a tuple equal to the number of
            context sets to use different values for each set. Set to a single
            value to use the same value for all context sets. Defaults to
            ``1 / internal_density``.
        encoder_scales_learnable (bool, optional):
            Whether the encoder SetConv length scale(s) are learnable.
            Defaults to ``False``.
        decoder_scale (float, optional):
            Initial value for the length scale of the set convolution in the
            decoder. Defaults to ``1 / internal_density``.
        decoder_scale_learnable (bool, optional):
            Whether the decoder SetConv length scale(s) are learnable. Defaults
            to ``False``.
        aux_t_mlp_layers (tuple[int], optional):
            Widths of the layers of the MLP for the target-specific auxiliary
            variable. Defaults to three layers of width 128.
        epsilon (float, optional):
            Epsilon added by the set convolutions before dividing by the
            density channel. Defaults to ``1e-2``.

    Returns:
        :class:`.model.Model`:
            ConvNP model.

    Raises:
        NotImplementedError
            If specified backend has no default dtype.
    """
    if likelihood == "cnp":
        likelihood = "het"
    elif likelihood == "gnp":
        likelihood = "lowrank"
    elif likelihood == "cnp-spikes-beta":
        likelihood = "spikes-beta"
    elif likelihood == "cnp-bernoulli-gamma":
        likelihood = "bernoulli-gamma"

    # Log the call signature for `construct_convgnp`
    config = dict(locals())

    if backend.str == "torch":
        import torch

        dtype = torch.float32
    elif backend.str == "tf":
        import tensorflow as tf

        dtype = tf.float32
    else:
        raise NotImplementedError(f"Backend {backend.str} has no default dtype.")

    neural_process = backend.nps.construct_convgnp(
        dim_x=dim_x,
        dim_yc=dim_yc,
        dim_yt=dim_yt,
        dim_aux_t=dim_aux_t,
        dim_lv=dim_lv,
        likelihood=likelihood,
        conv_arch=conv_arch,
        unet_channels=tuple(unet_channels),
        unet_resize_convs=unet_resize_convs,
        unet_resize_conv_interp_method=unet_resize_conv_interp_method,
        aux_t_mlp_layers=aux_t_mlp_layers,
        unet_kernels=unet_kernels,
        # Use a stride of 1 for the first layer and 2 for all other layers
        unet_strides=(1, *(2,) * (len(unet_channels) - 1)),
        points_per_unit=internal_density,
        encoder_scales=encoder_scales,
        encoder_scales_learnable=encoder_scales_learnable,
        decoder_scale=decoder_scale,
        decoder_scale_learnable=decoder_scale_learnable,
        num_basis_functions=num_basis_functions,
        epsilon=epsilon,
        dtype=dtype,
    )

    return neural_process, config


def compute_encoding_tensor(model, task: Task):
    """Compute the encoding tensor for a given task.

    Args:
        model (...):
            Model object.
        task (:class:`~.data.task.Task`):
            Task object containing context and target sets.

    Returns:
        encoding : :class:`numpy:numpy.ndarray`
            Encoding tensor? #TODO
    """
    neural_process_encoder = backend.nps.Model(model.model.encoder, lambda x: x)
    task = model.modify_task(task)
    encoding = B.to_numpy(run_nps_model(neural_process_encoder, task))
    return encoding
