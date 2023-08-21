from deepsensor.data.loader import TaskLoader

import scipy

import numpy as np
import pandas as pd
import xarray as xr

from typing import Union


def compute_xarray_data_resolution(ds: Union[xr.DataArray, xr.Dataset]) -> float:
    """Computes the resolution of an xarray object with coordinates x1 and x2

    Args:
        ds (Union[xr.DataArray, xr.Dataset]): Xarray object with coordinates x1 and x2.

    Returns:
        data_resolution (float): Resolution of the data (in spatial units).
    """
    x1_res = np.abs(np.mean(np.diff(ds["x1"])))
    x2_res = np.abs(np.mean(np.diff(ds["x2"])))
    data_resolution = np.mean([x1_res, x2_res])
    return data_resolution


def compute_pandas_data_resolution(
    df: Union[pd.DataFrame, pd.Series], n_times_samples: int = 1000, percentile: int = 5
) -> float:
    """Approximates the resolution of non-gridded pandas data with indexes time, x1, and x2.

    The resolution is approximated from a random selection of dates by, for each date:
     - computing the shortest distance to another observation for each observation
     - taking the 5th percentile (by default) to avoid outliers from the shortest distances
    Then, the mean of these resolutions is returned.

    Args:
        df (Union[pd.DataFrame, pd.Series]): Dataframe or series with indexes time, x1, and x2.
        n_times_samples (int, optional): Number of dates to sample. Defaults to 1000. If "all",
            all dates are used.
        percentile (int, optional): Percentile of pairwise distances for computing the resolution
            of each time slice. Defaults to 5.

    Returns:
        data_resolution (float): Resolution of the data (in spatial units).
    """
    dates = df.index.get_level_values("time").unique()

    if len(dates) > n_times_samples:
        rng = np.random.default_rng(42)
        dates = rng.choice(dates, size=n_times_samples, replace=False)

    data_resolutions = []
    df = df.reset_index().set_index("time")
    for time in dates:
        df_t = df.loc[time]
        X = df_t[["x1", "x2"]].values  # (N, 2) array of coordinates
        X_unique = np.unique(X, axis=0)  # (N_unique, 2) array of unique coordinates

        pairwise_distances = scipy.spatial.distance.cdist(X_unique, X_unique)
        percentile_distances_without_self = np.ma.masked_equal(pairwise_distances, 0)

        # Compute the closest distance from each station to each other station
        closest_distances = np.min(percentile_distances_without_self, axis=1)
        data_resolution = np.percentile(closest_distances, percentile)

        data_resolutions.append(data_resolution)

    return np.mean(data_resolutions)


def gen_ppu(task_loader: TaskLoader) -> int:
    """Computes data-informed settings for the model's internal discretisation density (ppu, points per unit)

    Loops over all context and target variables in the `TaskLoader` and computes the data resolution
    for each. The model ppu is then set to the maximum data ppu.
    - Xarray: data resolution is computed from the mean of the x1 and x2 coordinate resolutions
    - Pandas: data resolution is approximated using the distances between observations

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
                var, n_times_samples=1000, percentile=5
            )
        else:
            raise ValueError(f"Unknown context input type: {type(var)}")
        data_ppu = int(1 / data_resolution)
        data_ppus.append(data_ppu)

    model_ppu = int(max(data_ppus))
    return model_ppu


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
