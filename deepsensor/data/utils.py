from typing import Union

import numpy as np
import pandas as pd
import scipy
import xarray as xr


def construct_x1x2_ds(gridded_ds):
    """
    Construct an xr.Dataset containing two vars, where each var is a 2D gridded channel whose
    values contain the x_1 and x_2 coordinate values, respectively.
    """
    X1, X2 = np.meshgrid(gridded_ds.x1, gridded_ds.x2, indexing="ij")
    ds = xr.Dataset(
        coords={"x1": gridded_ds.x1, "x2": gridded_ds.x2},
        data_vars={"x1_arr": (("x1", "x2"), X1), "x2_arr": (("x1", "x2"), X2)},
    )
    return ds


def construct_circ_time_ds(dates, freq):
    """
    Return an xr.Dataset containing a circular variable for time. The `freq`
    entry dictates the frequency of cycling of the circular variable. E.g.:
        - 'H': cycles once per day at hourly intervals
        - 'D': cycles once per year at daily intervals
        - 'M': cycles once per year at monthly intervals
    """
    if freq == "D":
        time_var = dates.dayofyear
        mod = 365.25
    elif freq == "H":
        time_var = dates.hour
        mod = 24
    elif freq == "M":
        time_var = dates.month
        mod = 12
    else:
        raise ValueError(
            "Circular time variable not implemented " "for this frequency."
        )

    cos_time = np.cos(2 * np.pi * time_var / mod)
    sin_time = np.sin(2 * np.pi * time_var / mod)

    ds = xr.Dataset(
        coords={"time": dates},
        data_vars={
            f"cos_{freq}": ("time", cos_time),
            f"sin_{freq}": ("time", sin_time),
        },
    )
    return ds


def compute_xarray_data_resolution(ds: Union[xr.DataArray, xr.Dataset]) -> float:
    """Computes the resolution of an xarray object with coordinates x1 and x2

    The data resolution is the finer of the two coordinate resolutions (x1 and x2). For example, if
    x1 has a resolution of 0.1 degrees and x2 has a resolution of 0.2 degrees, the data resolution
    returned will be 0.1 degrees.

    Args:
        ds (Union[xr.DataArray, xr.Dataset]): Xarray object with coordinates x1 and x2.

    Returns:
        data_resolution (float): Resolution of the data (in spatial units, e.g. 0.1 degrees).
    """
    x1_res = np.abs(np.mean(np.diff(ds["x1"])))
    x2_res = np.abs(np.mean(np.diff(ds["x2"])))
    data_resolution = np.min([x1_res, x2_res])
    return data_resolution


def compute_pandas_data_resolution(
    df: Union[pd.DataFrame, pd.Series], n_times: int = 1000, percentile: int = 5
) -> float:
    """Approximates the resolution of non-gridded pandas data with indexes time, x1, and x2.

    The resolution is approximated as the Nth percentile of the distances between neighbouring
    observations, possibly using a subset of the dates in the data. The default is to use 1000
    dates (or all dates if there are fewer than 1000) and to use the 5th percentile. This means
    that the resolution is the distance between the closest 5% of neighbouring observations.

    Args:
        df (Union[pd.DataFrame, pd.Series]): Dataframe or series with indexes time, x1, and x2.
        n_times (int, optional): Number of dates to sample. Defaults to 1000. If "all", all dates
            are used.
        percentile (int, optional): Percentile of pairwise distances for computing the resolution.
            Defaults to 5.

    Returns:
        data_resolution (float): Resolution of the data (in spatial units, e.g. 0.1 degrees).
    """
    dates = df.index.get_level_values("time").unique()

    if n_times != "all" and len(dates) > n_times:
        rng = np.random.default_rng(42)
        dates = rng.choice(dates, size=n_times, replace=False)

    closest_distances = []
    df = df.reset_index().set_index("time")
    for time in dates:
        df_t = df.loc[[time]]
        X = df_t[["x1", "x2"]].values  # (N, 2) array of coordinates
        if X.shape[0] < 2:
            # Skip this time if there are fewer than 2 stationS
            continue
        X_unique = np.unique(X, axis=0)  # (N_unique, 2) array of unique coordinates

        pairwise_distances = scipy.spatial.distance.cdist(X_unique, X_unique)
        percentile_distances_without_self = np.ma.masked_equal(pairwise_distances, 0)

        # Compute the closest distance from each station to each other station
        closest_distances_t = np.min(percentile_distances_without_self, axis=1)
        closest_distances.extend(closest_distances_t)

    data_resolution = np.percentile(closest_distances, percentile)
    return data_resolution
