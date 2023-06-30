import numpy as np
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
