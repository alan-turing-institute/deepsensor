from deepsensor.data.loader import TaskLoader

import numpy as np
import pandas as pd
import xarray as xr

from deepsensor.data.utils import (
    compute_xarray_data_resolution,
    compute_pandas_data_resolution,
)


def gen_ppu(task_loader: TaskLoader) -> int:
    """Computes data-informed settings for the model's internal grid density (ppu, points per unit)

    Loops over all context and target variables in the `TaskLoader` and computes the data resolution
    for each. The model ppu is then set to the maximum data ppu.

    Args:
        task_loader (TaskLoader): TaskLoader object containing context and target sets.

    Returns:
        model_ppu (int): Model ppu (points per unit), i.e. the number of points per unit of input space.
    """
    # List of data resolutions for each context/target variable (in points-per-unit)
    data_ppus = []
    for var in [*task_loader.context, *task_loader.target]:
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            # Gridded variable: use data resolution
            data_resolution = compute_xarray_data_resolution(var)
        elif isinstance(var, (pd.DataFrame, pd.Series)):
            # Point-based variable: calculate ppu based on pairwise distances between observations
            data_resolution = compute_pandas_data_resolution(
                var, n_times=1000, percentile=5
            )
        else:
            raise ValueError(f"Unknown context input type: {type(var)}")
        data_ppu = int(1 / data_resolution)
        data_ppus.append(data_ppu)

    model_ppu = int(max(data_ppus))
    return model_ppu


def gen_decoder_scale(model_ppu: int) -> float:
    """Computes informed setting for the decoder SetConv scale

    This sets the length scale of the Gaussian basis functions used interpolate from the
    model's internal grid to the target locations.

    The decoder scale should be as small as possible given the model's internal grid.
    The value chosen is 1 / model_ppu (i.e. the length scale is equal to the model's internal
    grid spacing).
    """
    return 1 / model_ppu


def gen_encoder_scales(model_ppu: int, task_loader: TaskLoader) -> list[float]:
    """Computes data-informed settings for the encoder SetConv scale for each context set

    This sets the length scale of the Gaussian basis functions used to encode the context sets.

    For off-grid station data, the scale should be as small as possible given the model's
    internal grid density (ppu, points per unit). The value chosen is 0.5 / model_ppu
    (i.e. half the model's internal resolution).

    For gridded data, the scale should be such that the functional representation smoothly
    interpolates the data. This is determined by computing the *data resolution* (the distance
    between the nearest two data points) for each context variable. The encoder scale is then
    set to 0.5 * data_resolution.

    Args:
        model_ppu (int): Model ppu (points per unit), i.e. the number of points per unit of input space.
        task_loader (TaskLoader): TaskLoader object containing context and target sets.

    Returns:
        encoder_scales (list[float]): List of encoder scales for each context set.
    """
    encoder_scales = []
    for var in task_loader.context:
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            encoder_scale = 0.5 * compute_xarray_data_resolution(var)
        elif isinstance(var, (pd.DataFrame, pd.Series)):
            encoder_scale = 0.5 / model_ppu
        else:
            raise ValueError(f"Unknown context input type: {type(var)}")
        encoder_scales.append(encoder_scale)

    if task_loader.aux_at_contexts:
        # Add encoder scale for the final auxiliary-at-contexts context set: use smallest possible
        # scale within model discretisation
        encoder_scales.append(0.5 / model_ppu)

    return encoder_scales
