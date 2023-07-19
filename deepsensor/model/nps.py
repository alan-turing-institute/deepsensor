from deepsensor import backend
import lab as B

from deepsensor.data.task import Task


def convert_task_to_nps_args(task: Task):
    """Infer & build model call signature from `task` dict

    TODO move to ConvNP class?
    """

    context_data = list(zip(task["X_c"], task["Y_c"]))

    if len(task["X_t"]) == 1 and len(task["Y_t"]) == 1:
        # Single target set
        xt = task["X_t"][0]
        yt = task["Y_t"][0]
    elif len(task["X_t"]) > 1 and len(task["Y_t"]) > 1:
        # Multiple target sets, different target locations
        xt = backend.nps.AggregateInput(*[(xt, i) for i, xt in enumerate(task["X_t"])])
        yt = backend.nps.Aggregate(*task["Y_t"])
    elif len(task["X_t"]) == 1 and len(task["Y_t"]) > 1:
        # Multiple target sets, same target locations
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


def run_nps_model(neural_process, task, n_samples=None, requires_grad=False):
    """Run `neuralprocesses` model"""
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


def run_nps_model_ar(neural_process, task, num_samples=1):
    """Run `neural_process` in AR mode"""
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
    dim_x=2,
    dim_yc=1,
    dim_yt=1,
    dim_aux_t=None,
    dim_lv=0,
    conv_arch="unet",
    unet_channels=(64, 64, 64, 64),
    unet_resize_convs=True,
    unet_resize_conv_interp_method="bilinear",
    aux_t_mlp_layers=None,
    likelihood="cnp",
    unet_kernels=5,
    points_per_unit=100,
    encoder_scales=1 / 100,
    encoder_scales_learnable=False,
    decoder_scale=1 / 100,
    decoder_scale_learnable=False,
    num_basis_functions=64,
    epsilon=1e-2,
):
    """Construct a `neuralprocesses` ConvNP model

    See: https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses/architectures/convgnp.py

    Docstring below modified from `neuralprocesses`. If more kwargs are needed, they must be
    explicitly passed to `neuralprocesses` constructor (not currently safe to use `**kwargs` here).

    Args:
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
        points_per_unit (int, optional): Density of the internal discretisation.
            Defaults to 100.
        likelihood (str, optional): Likelihood. Must be one of `"cnp"` (equivalently `"het"`),
            `"gnp"` (equivalently `"lowrank"`), or `"cnp-spikes-beta"` (equivalently `"spikes-beta"`).
            Defaults to `"cnp"`.
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

    Returns:
        :class:`.model.Model`: ConvNP model.
    """
    if likelihood == "cnp":
        likelihood = "het"
    elif likelihood == "gnp":
        likelihood = "lowrank"
    elif likelihood == "cnp-spikes-beta":
        likelihood = "spikes-beta"

    # Use a stride of 1 for the first layer and 2 for all other layers
    unet_strides = (1, *(2,) * (len(unet_channels) - 1))

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
        conv_arch=conv_arch,
        unet_channels=unet_channels,
        unet_resize_convs=unet_resize_convs,
        unet_resize_conv_interp_method=unet_resize_conv_interp_method,
        aux_t_mlp_layers=aux_t_mlp_layers,
        likelihood=likelihood,
        unet_kernels=unet_kernels,
        unet_strides=unet_strides,
        points_per_unit=points_per_unit,
        encoder_scales=encoder_scales,
        encoder_scales_learnable=encoder_scales_learnable,
        decoder_scale=decoder_scale,
        decoder_scale_learnable=decoder_scale_learnable,
        num_basis_functions=num_basis_functions,
        epsilon=epsilon,
        dtype=dtype,
    )
    return neural_process


def compute_encoding_tensor(model, task: Task):
    neural_process_encoder = backend.nps.Model(model.model.encoder, lambda x: x)
    task = model.check_task(task)
    encoding = B.to_numpy(run_nps_model(neural_process_encoder, task))
    return encoding
