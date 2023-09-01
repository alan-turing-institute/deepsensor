import numpy as np
import pandas as pd
import xarray as xr


def gen_random_data_xr(coords: dict, dims: list = None, data_vars: list = None):
    """
    Generate random xarray data.

    Parameters
    ----------
    coords : dict
        Coordinates of the data.
    dims : list, optional
        Dimensions of the data. Defaults to None. If None, dims is inferred
        from coords. This arg can be used to change the order of the
        dimensions.
    data_vars : list, optional
        Data variables. Defaults to None. If None, variable is an
        xr.DataArray. If not None, variable is an xr.Dataset containing the
        data_vars.

    Returns
    -------
    da : Union[xr.DataArray, xr.Dataset]
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
    """
    Generate random pandas data.

    Parameters
    ----------
    coords : dict
        Coordinates of the data. This will be used to construct a MultiIndex
        using pd.MultiIndex.from_product.
    dims : list, optional
        Dimensions of the data. Defaults to None. If None, dims is inferred
        from coords. This arg can be used to change the order of the MultiIndex.
    cols : list, optional
        Columns of the data. Defaults to None. If None, generate a pd.Series
        with an arbitrary name. If not None, cols is used to construct a
        pd.DataFrame.

    Returns
    -------
    df : Union[pd.Series, pd.DataFrame]
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
