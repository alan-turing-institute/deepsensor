from deepsensor.data.loader import TaskLoader

import numpy as np
import pandas as pd
import xarray as xr

from deepsensor.data.utils import (
    compute_xarray_data_resolution,
    compute_pandas_data_resolution,
)

from typing import List


def compute_greatest_data_density(task_loader: TaskLoader) -> int:
    """Computes data-informed settings for the model's internal grid density (ppu,
    points per unit).

    Loops over all context and target variables in the ``TaskLoader`` and
    computes the data resolution for each. The model ppu is then set to the
    maximum data ppu.

    Args:
        task_loader (:class:`~.data.loader.TaskLoader`):
            TaskLoader object containing context and target sets.

    Returns:
        max_density (int):
            The maximum data density (ppu) across all context and target
            variables, where 'density' is the number of points per unit of
            input space (in both spatial dimensions).
    """
    # List of data resolutions for each context/target variable (in points-per-unit)
    data_densities = []
    for var in [*task_loader.context, *task_loader.target]:
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            # Gridded variable: use data resolution
            data_resolution = compute_xarray_data_resolution(var)
        elif isinstance(var, (pd.DataFrame, pd.Series)):
            # Point-based variable: calculate density based on pairwise distances between observations
            data_resolution = compute_pandas_data_resolution(
                var, n_times=1000, percentile=5
            )
        else:
            raise ValueError(f"Unknown context input type: {type(var)}")
        data_density = int(1 / data_resolution)
        data_densities.append(data_density)
    max_density = int(max(data_densities))
    return max_density


def gen_decoder_scale(model_ppu: int) -> float:
    """Computes informed setting for the decoder SetConv scale.

    This sets the length scale of the Gaussian basis functions used interpolate
    from the model's internal grid to the target locations.

    The decoder scale should be as small as possible given the model's
    internal grid. The value chosen is 1 / model_ppu (i.e. the length scale is
    equal to the model's internal grid spacing).

    Args:
        model_ppu (int):
            Model ppu (points per unit), i.e. the number of points per unit of
            input space.

    Returns:
        float: Decoder scale.
    """
    return 1 / model_ppu


def gen_encoder_scales(model_ppu: int, task_loader: TaskLoader) -> List[float]:
    """Computes data-informed settings for the encoder SetConv scale for each
    context set.

    This sets the length scale of the Gaussian basis functions used to encode
    the context sets.

    For off-grid station data, the scale should be as small as possible given
    the model's internal grid density (ppu, points per unit). The value chosen
    is 0.5 / model_ppu (i.e. half the model's internal resolution).

    For gridded data, the scale should be such that the functional
    representation smoothly interpolates the data. This is determined by
    computing the *data resolution* (the distance between the nearest two data
    points) for each context variable. The encoder scale is then set to 0.5 *
    data_resolution.

    Args:
        model_ppu (int):
            Model ppu (points per unit), i.e. the number of points per unit of
            input space.
        task_loader (:class:`~.data.loader.TaskLoader`):
            TaskLoader object containing context and target sets.

    Returns:
        list[float]: List of encoder scales for each context set.
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
