from deepsensor import backend
import numpy as np


def convert_task_to_nps_args(task):
    """Infer & build model call signature from `task` dict

    TODO move to ConvNP class?
    """

    context_data = list(zip(task["X_c"], task["Y_c"]))

    # context_data = [nps.mask.merge_contexts(ctx, multiple=5000) for ctx in context_data]
    # context_data[0] = nps.mask.merge_contexts(context_data[0], multiple=5000)  # TEMP not converting gridded

    target_set_idx = 0
    # TEMP: assume just one target set and just use the first entry from the lists of target data
    xt = task["X_t"][target_set_idx]
    yt = task["Y_t"][target_set_idx]
    # xt = tf.cast(tf.convert_to_tensor(xt), tf.float32)

    # TODO implement this and test downstream functionality
    # xt = nps.AggregateInput(*[(xt, i) for i, xt in enumerate(task['X_t'])])
    # yt = nps.Aggregate(*[yt for yt in enumerate(task['Y_t'])])

    # Assume one target set, convert to tf.Tensor and AggregateInput for AR
    #   sampling
    # xt = tf.cast(tf.convert_to_tensor(xt), tf.float32)
    # xt = nps.aggregate.AggregateInput(
    #     (xt, 0),
    # )

    model_kwargs = {}
    if "Y_target_auxiliary" in task.keys():
        model_kwargs["aux_t"] = task["Y_target_auxiliary"]

    return context_data, xt, yt, model_kwargs


def run_nps_model(neural_process, task, n_samples=None):
    """Run `neuralprocesses` model"""
    context_data, xt, yt, model_kwargs = convert_task_to_nps_args(task)
    dist = neural_process(context_data, xt, **model_kwargs, num_samples=n_samples)
    return dist


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
    likelihood="lowrank",
    unet_kernels=5,
    points_per_unit=100,
    # By default use a different setconv length scale for each context set
    encoder_scales=1 / 100,
    encoder_scales_learnable=False,
    decoder_scale=1 / 100,
    decoder_scale_learnable=False,
    num_basis_functions=64,
    epsilon=1e-2,
):
    """Construct a `neuralprocesses` ConvNP model"""

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


def compute_encoding_tensor(model, task):
    neural_process_encoder = backend.nps.Model(model.model.encoder, lambda x: x)
    task = model.check_task(task)
    # encoding = run_nps_model(neural_process_encoder, task).numpy()[0]
    encoding = run_nps_model(neural_process_encoder, task)[0]
    return encoding
