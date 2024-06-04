import numpy as np
import pandas as pd
import xarray as xr

from typing import Union, Optional


def gen_random_data_xr(
    coords: dict, dims: Optional[list] = None, data_vars: Optional[list] = None
):
    """Generate random xarray data.

    Args:
        coords (dict):
            Coordinates of the data.
        dims (list, optional):
            Dimensions of the data. Defaults to None. If None, dims is inferred
            from coords. This arg can be used to change the order of the
            dimensions.
        data_vars (list, optional):
            Data variables. Defaults to None. If None, variable is an
            :class:`xarray.DataArray`. If not None, variable is an
            :class:`xarray.Dataset` containing the data_vars.

    Returns:
        da (:class:`xarray.DataArray` | :class:`xarray.Dataset`):
            Random xarray data.
    """
    if dims is None:
        shape = tuple([len(coords[dim]) for dim in coords])
    else:
        shape = tuple([len(coords[dim]) for dim in dims])
    data = np.random.rand(*shape)
    if data_vars is None:
        name = "var"
        da = xr.DataArray(data, coords=coords, name=name)
    else:
        data = {var: xr.DataArray(data, coords=coords) for var in data_vars}
        da = xr.Dataset(data, coords=coords)
    return da


def gen_random_data_pandas(coords: dict, dims: list = None, cols: list = None):
    """Generate random pandas data.

    Args:
        coords (dict):
            Coordinates of the data. This will be used to construct a
            MultiIndex using pandas.MultiIndex.from_product.
        dims (list, optional):
            Dimensions of the data. Defaults to None. If None, dims is inferred
            from coords. This arg can be used to change the order of the
            MultiIndex.
        cols (list, optional):
            Columns of the data. Defaults to None. If None, generate a
            :class:`pandas.Series` with an arbitrary name. If not None, cols is
            used to construct a :class:`pandas.DataFrame`.

    Returns:
        :class:`pandas.Series` | :class:`pandas.DataFrame`
            Random pandas data.
    """
    if dims is None:
        dims = list(coords.keys())
    mi = pd.MultiIndex.from_product([coords[dim] for dim in dims], names=dims)
    if cols is None:
        name = "var"
        df = pd.Series(index=mi, name=name)
    else:
        df = pd.DataFrame(index=mi, columns=cols)
    df[:] = np.random.rand(*df.shape)
    return df


def assert_allclose_pd(
    df1: Union[pd.DataFrame, pd.Series], df2: Union[pd.DataFrame, pd.Series]
):
    if isinstance(df1, pd.Series):
        df1 = df1.to_frame()
    if isinstance(df2, pd.Series):
        df2 = df2.to_frame()
    try:
        pd.testing.assert_frame_equal(df1, df2)
    except AssertionError:
        return False
    return True


def assert_allclose_xr(
    da1: Union[xr.DataArray, xr.Dataset], da2: Union[xr.DataArray, xr.Dataset]
):
    try:
        xr.testing.assert_allclose(da1, da2)
    except AssertionError:
        return False
    return True
