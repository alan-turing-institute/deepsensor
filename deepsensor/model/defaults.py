from deepsensor.data.loader import TaskLoader

import numpy as np
import pandas as pd
import xarray as xr


def gen_ppu(task_loader: TaskLoader) -> int:
    """Computes data-informed settings for the model's internal discretisation density (ppu, points per unit)

    For gridded data, the ppu should be as small as possible given the data's resolution. The value
    chosen is 1.2 * data_ppu, where data_ppu is the number of data points per unit.

    For off-grid station data, the ppu should be as large as possible so that the model can
    represent the data as accurately as possible. The value chosen is 300.
    """
    max_ppu = 0
    for var in [*task_loader.context, *task_loader.target]:
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            # Gridded variable: use data ppu
            x1_res = np.abs(np.mean(np.diff(var["x1"])))
            x2_res = np.abs(np.mean(np.diff(var["x2"])))
            data_ppu = 1 / np.mean([x1_res, x2_res])
            max_ppu = max(max_ppu, data_ppu * 1.2)  # Add 20% margin
        elif isinstance(var, (pd.DataFrame, pd.Series)):
            # Point-based variable: make ppu as large as possible
            # TODO: Consider choosing based on shortest distance between points, perhaps with some
            # check for outliers causing excessively large ppu
            max_ppu = 300
        else:
            raise ValueError(f"Unknown context input type: {type(var)}")

    return int(max_ppu)


def gen_decoder_scale(model_ppu: int) -> float:
    """Computes informed setting for the decoder SetConv scale

    The decoder scale should be as small as possible given the model's internal discretisation.
    The value chosen is 1 / model_ppu.
    """
    return 1 / model_ppu


def gen_encoder_scales(model_ppu: int, task_loader: TaskLoader) -> list:
    """Computes data-informed settings for the encoder SetConv scale for each context set

    For off-grid station data, the scale should be as small as possible given the model's
    internal discretisation density (ppu, points per unit). The value chosen is 0.5 / model_ppu.

    For gridded data, the scale should be such that the functional representation smoothly
    interpolates the data. This is determined by computing the *data ppu* (number of gridded
    data points in a 1x1 square of normalised input space). The value chosen is then 0.5 / data ppu.
    """
    encoder_scales = []
    for var in task_loader.context:
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            # Gridded variable: use data ppu
            x1_res = np.abs(np.mean(np.diff(var["x1"])))
            x2_res = np.abs(np.mean(np.diff(var["x2"])))
            data_ppu = 1 / np.mean([x1_res, x2_res])
            encoder_scale = 0.5 / data_ppu
        elif isinstance(var, (pd.DataFrame, pd.Series)):
            # Point-based variable: use smallest possible scale within model discretisation
            encoder_scale = 0.5 / model_ppu
        else:
            raise ValueError(f"Unknown context input type: {type(var)}")
        encoder_scales.append(encoder_scale)

    return encoder_scales
