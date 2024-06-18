import numpy as np
import os
import json

import warnings
import xarray as xr
import pandas as pd

import pprint

from copy import deepcopy

from plum import dispatch
from typing import Union, Optional, List


class DataProcessor:
    """Normalise xarray and pandas data for use in deepsensor models.

    Args:
        folder (str, optional):
            Folder to load normalisation params from. Defaults to None.
        x1_name (str, optional):
            Name of first spatial coord (e.g. "lat"). Defaults to "x1".
        x2_name (str, optional):
            Name of second spatial coord (e.g. "lon"). Defaults to "x2".
        x1_map (tuple, optional):
            2-tuple of raw x1 coords to linearly map to (0, 1),
            respectively. Defaults to (0, 1) (i.e. no normalisation).
        x2_map (tuple, optional):
            2-tuple of raw x2 coords to linearly map to (0, 1),
            respectively. Defaults to (0, 1) (i.e. no normalisation).
        deepcopy (bool, optional):
            Whether to make a deepcopy of raw data to ensure it is not
            changed by reference when normalising. Defaults to True.
        verbose (bool, optional):
            Whether to print verbose output. Defaults to False.
    """

    config_fname = "data_processor_config.json"

    def __init__(
        self,
        folder: Union[str, None] = None,
        time_name: str = "time",
        x1_name: str = "x1",
        x2_name: str = "x2",
        x1_map: Union[tuple, None] = None,
        x2_map: Union[tuple, None] = None,
        deepcopy: bool = True,
        verbose: bool = False,
    ):
        if folder is not None:
            fpath = os.path.join(folder, self.config_fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Could not find DataProcessor config file {fpath}"
                )
            with open(fpath, "r") as f:
                self.config = json.load(f)
                self.config["coords"]["x1"]["map"] = tuple(
                    self.config["coords"]["x1"]["map"]
                )
                self.config["coords"]["x2"]["map"] = tuple(
                    self.config["coords"]["x2"]["map"]
                )

            self.x1_name = self.config["coords"]["x1"]["name"]
            self.x2_name = self.config["coords"]["x2"]["name"]
            self.x1_map = self.config["coords"]["x1"]["map"]
            self.x2_map = self.config["coords"]["x2"]["map"]
        else:
            self.config = {}
            self.x1_name = x1_name
            self.x2_name = x2_name
            self.x1_map = x1_map
            self.x2_map = x2_map

            # rewrite below more concisely
            if self.x1_map is None and not self.x2_map is None:
                raise ValueError("Must provide both x1_map and x2_map, or neither.")
            elif not self.x1_map is None and self.x2_map is None:
                raise ValueError("Must provide both x1_map and x2_map, or neither.")
            elif not self.x1_map is None and not self.x2_map is None:
                x1_map, x2_map = self._validate_coord_mappings(x1_map, x2_map)

            if "coords" not in self.config:
                # Add coordinate normalisation info to config
                self.set_coord_params(time_name, x1_name, x1_map, x2_name, x2_map)

        self.raw_spatial_coord_names = [
            self.config["coords"][coord]["name"] for coord in ["x1", "x2"]
        ]

        self.deepcopy = deepcopy
        self.verbose = verbose

        # List of valid normalisation method names
        self.valid_methods = ["mean_std", "min_max"]

    def save(self, folder: str):
        """Save DataProcessor config to JSON in `folder`."""
        os.makedirs(folder, exist_ok=True)
        fpath = os.path.join(folder, self.config_fname)
        with open(fpath, "w") as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

    def _validate_coord_mappings(self, x1_map, x2_map):
        """Ensure the maps are valid and of appropriate types."""
        try:
            x1_map = (float(x1_map[0]), float(x1_map[1]))
            x2_map = (float(x2_map[0]), float(x2_map[1]))
        except:
            raise TypeError(
                "Provided coordinate mappings can't be cast to 2D Tuple[float]"
            )

        # Check that map is not two of the same number
        if np.diff(x1_map) == 0:
            raise ValueError(
                f"x1_map must be a 2-tuple of different numbers, not {x1_map}"
            )
        if np.diff(x2_map) == 0:
            raise ValueError(
                f"x2_map must be a 2-tuple of different numbers, not {x2_map}"
            )
        if np.diff(x1_map) != np.diff(x2_map):
            warnings.warn(
                f"x1_map={x1_map} and x2_map={x2_map} have different ranges ({float(np.diff(x1_map))} "
                f"and {float(np.diff(x2_map))}, respectively). "
                "This can lead to stretching/squashing of data, which may "
                "impact model performance.",
                UserWarning,
            )

        return x1_map, x2_map

    def _validate_xr(self, data: Union[xr.DataArray, xr.Dataset]):
        def _validate_da(da: xr.DataArray):
            coord_names = [
                self.config["coords"][coord]["name"] for coord in ["time", "x1", "x2"]
            ]
            if coord_names[0] not in da.dims:
                # We don't have a time dimension.
                coord_names = coord_names[1:]
            if list(da.dims) != coord_names:
                raise ValueError(
                    f"Dimensions of {da.name} need to be {coord_names} but are {list(da.dims)}."
                )

        if isinstance(data, xr.DataArray):
            _validate_da(data)

        elif isinstance(data, xr.Dataset):
            for var_ID, da in data.data_vars.items():
                _validate_da(da)

    def _validate_pandas(self, df: Union[pd.DataFrame, pd.Series]):
        coord_names = [
            self.config["coords"][coord]["name"] for coord in ["time", "x1", "x2"]
        ]

        if coord_names[0] not in df.index.names:
            # We don't have a time dimension.
            if list(df.index.names)[:2] != coord_names[1:]:
                raise ValueError(
                    f"Indexes need to start with {coord_names} or {coord_names[1:]} but are {df.index.names}."
                )
        else:
            # We have a time dimension.
            if list(df.index.names)[:3] != coord_names:
                raise ValueError(
                    f"Indexes need to start with {coord_names} or {coord_names[1:]} but are {df.index.names}."
                )

    def __str__(self):
        s = "DataProcessor with normalisation params:\n"
        s += pprint.pformat(self.config)
        return s

    @classmethod
    def load_dask(cls, data: Union[xr.DataArray, xr.Dataset]):
        """Load dask data into memory.

        Args:
            data (:class:`xarray.DataArray` | :class:`xarray.Dataset`):
                Description of the parameter.

        Returns:
            [Type and description of the returned value(s) needed.]
        """
        if isinstance(data, xr.DataArray):
            data.load()
        elif isinstance(data, xr.Dataset):
            data.load()
        return data

    def set_coord_params(self, time_name, x1_name, x1_map, x2_name, x2_map) -> None:
        """Set coordinate normalisation params.

        Args:
            time_name:
                [Type] Description needed.
            x1_name:
                [Type] Description needed.
            x1_map:
                [Type] Description needed.
            x2_name:
                [Type] Description needed.
            x2_map:
                [Type] Description needed.

        Returns:
            None.
        """
        self.config["coords"] = {}
        self.config["coords"]["time"] = {"name": time_name}
        self.config["coords"]["x1"] = {}
        self.config["coords"]["x2"] = {}
        self.config["coords"]["x1"]["name"] = x1_name
        self.config["coords"]["x1"]["map"] = x1_map
        self.config["coords"]["x2"]["name"] = x2_name
        self.config["coords"]["x2"]["map"] = x2_map

    def check_params_computed(self, var_ID, method) -> bool:
        """Check if normalisation params computed for a given variable.

        Args:
            var_ID:
                [Type] Description needed.
            method:
                [Type] Description needed.

        Returns:
            bool:
                Whether normalisation params are computed for a given variable.
        """
        if (
            var_ID in self.config
            and self.config[var_ID]["method"] == method
            and "params" in self.config[var_ID]
        ):
            return True

        return False

    def add_to_config(self, var_ID, **kwargs):
        """Add `kwargs` to `config` dict for variable `var_ID`."""
        self.config[var_ID] = kwargs

    def get_config(self, var_ID, data, method=None):
        """Get pre-computed normalisation params or compute them for variable
        ``var_ID``.

        .. note::
            TODO do we need to pass var_ID? Can we just use the name of data?

        Args:
            var_ID:
                [Type] Description needed.
            data:
                [Type] Description needed.
            method (optional):
                [Type] Description needed. Defaults to None.

        Returns:
            [Type]:
                Description of the returned value(s) needed.
        """
        if method not in self.valid_methods:
            raise ValueError(
                f"Method {method} not recognised. Must be one of {self.valid_methods}"
            )

        if self.check_params_computed(var_ID, method):
            # Already have "params" in config with `"method": method` - load them
            params = self.config[var_ID]["params"]
        else:
            # Params not computed - compute them now
            if self.verbose:
                print(
                    f"Normalisation params for {var_ID} not computed. Computing now... ",
                    end="",
                    flush=True,
                )
            DataProcessor.load_dask(data)
            if method == "mean_std":
                params = {"mean": float(data.mean()), "std": float(data.std())}
            elif method == "min_max":
                params = {"min": float(data.min()), "max": float(data.max())}
            if self.verbose:
                print(f"Done. {var_ID} {method} params={params}")
            self.add_to_config(
                var_ID,
                **{"method": method, "params": params},
            )
        return params

    def map_coord_array(self, coord_array: np.ndarray, unnorm: bool = False):
        """Normalise or unnormalise a coordinate array.

        Args:
            coord_array (:class:`numpy:numpy.ndarray`):
                Array of shape ``(2, N)`` containing coords.
            unnorm (bool, optional):
                Whether to unnormalise. Defaults to ``False``.

        Returns:
            [Type]:
                Description of the returned value(s) needed.
        """
        x1, x2 = self.map_x1_and_x2(coord_array[0], coord_array[1], unnorm=unnorm)
        new_coords = np.stack([x1, x2], axis=0)
        return new_coords

    def map_x1_and_x2(self, x1: np.ndarray, x2: np.ndarray, unnorm: bool = False):
        """Normalise or unnormalise spatial coords in an array.

        Args:
            x1 (:class:`numpy:numpy.ndarray`):
                Array of shape ``(N_x1,)`` containing spatial coords of x1.
            x2 (:class:`numpy:numpy.ndarray`):
                Array of shape ``(N_x2,)`` containing spatial coords of x2.
            unnorm (bool, optional):
                Whether to unnormalise. Defaults to ``False``.

        Returns:
            Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]:
                Normalised or unnormalised spatial coords of x1 and x2.
        """
        x11, x12 = self.config["coords"]["x1"]["map"]
        x21, x22 = self.config["coords"]["x2"]["map"]

        if not unnorm:
            new_coords_x1 = (x1 - x11) / (x12 - x11)
            new_coords_x2 = (x2 - x21) / (x22 - x21)
        else:
            new_coords_x1 = x1 * (x12 - x11) + x11
            new_coords_x2 = x2 * (x22 - x21) + x21

        return new_coords_x1, new_coords_x2

    def map_coords(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series],
        unnorm=False,
    ):
        """Normalise spatial coords in a pandas or xarray object.

        Args:
            data (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`pandas.DataFrame`, or :class:`pandas.Series`):
                [Description Needed]
            unnorm (bool, optional):
                [Description Needed]. Defaults to [Default Value].

        Returns:
            [Type]:
                [Description Needed]
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Reset index to get coords as columns
            indexes = list(data.index.names)
            data = data.reset_index()

        if unnorm:
            new_coord_IDs = [
                self.config["coords"][coord_ID]["name"]
                for coord_ID in ["time", "x1", "x2"]
            ]
            old_coord_IDs = ["time", "x1", "x2"]
        else:
            new_coord_IDs = ["time", "x1", "x2"]
            old_coord_IDs = [
                self.config["coords"][coord_ID]["name"]
                for coord_ID in ["time", "x1", "x2"]
            ]

        x1, x2 = (
            data[old_coord_IDs[1]],
            data[old_coord_IDs[2]],
        )

        # Infer x1 and x2 mappings from min/max of data coords if not provided by user
        if self.x1_map is None and self.x2_map is None:
            # Ensure scalings are the same for x1 and x2
            x1_range = x1.max() - x1.min()
            x2_range = x2.max() - x2.min()
            range = np.max([x1_range, x2_range])
            self.x1_map = (x1.min(), x1.min() + range)
            self.x2_map = (x2.min(), x2.min() + range)

            self.x1_map, self.x2_map = self._validate_coord_mappings(
                self.x1_map, self.x2_map
            )
            self.config["coords"]["x1"]["map"] = self.x1_map
            self.config["coords"]["x2"]["map"] = self.x2_map

            if self.verbose:
                print(
                    f"Inferring x1_map={self.x1_map} and x2_map={self.x2_map} from data min/max"
                )

        new_x1, new_x2 = self.map_x1_and_x2(x1, x2, unnorm=unnorm)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Drop old spatial coord columns *before* adding new ones, in case
            # the old name is already x1.
            data = data.drop(columns=old_coord_IDs[1:])
            # Add coords to dataframe
            data[new_coord_IDs[1]] = new_x1
            data[new_coord_IDs[2]] = new_x2

            if old_coord_IDs[0] in data.columns:
                # Rename time dimension.
                rename = {old_coord_IDs[0]: new_coord_IDs[0]}
                data = data.rename(rename, axis=1)
            else:
                # We don't have a time dimension.
                old_coord_IDs = old_coord_IDs[1:]
                new_coord_IDs = new_coord_IDs[1:]

        elif isinstance(data, (xr.DataArray, xr.Dataset)):
            data = data.assign_coords(
                {old_coord_IDs[1]: new_x1, old_coord_IDs[2]: new_x2}
            )

            if old_coord_IDs[0] not in data.dims:
                # We don't have a time dimension.
                old_coord_IDs = old_coord_IDs[1:]
                new_coord_IDs = new_coord_IDs[1:]

            # Rename all dimensions.
            rename = {
                old: new for old, new in zip(old_coord_IDs, new_coord_IDs) if old != new
            }
            data = data.rename(rename)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Set index back to original
            [indexes.remove(old_coord_ID) for old_coord_ID in old_coord_IDs]
            indexes = new_coord_IDs + indexes  # Put dims first
            data = data.set_index(indexes)

        return data

    def map_array(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series, np.ndarray],
        var_ID: str,
        method: Optional[str] = None,
        unnorm: bool = False,
        add_offset: bool = True,
    ):
        """Normalise or unnormalise the data values in an xarray, pandas, or
        numpy object.

        Args:
            data (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`pandas.DataFrame`, :class:`pandas.Series`, or :class:`numpy:numpy.ndarray`):
                [Description Needed]
            var_ID (str):
                [Description Needed]
            method (str, optional):
                [Description Needed]. Defaults to None.
            unnorm (bool, optional):
                [Description Needed]. Defaults to False.
            add_offset (bool, optional):
                [Description Needed]. Defaults to True.

        Returns:
            [Type]:
                [Description Needed]
        """
        if not unnorm and method is None:
            raise ValueError("Must provide `method` if normalising data.")
        elif unnorm and method is not None and self.config[var_ID]["method"] != method:
            # User has provided a different method to the one used for normalising
            raise ValueError(
                f"Variable '{var_ID}' has been normalised with method '{self.config[var_ID]['method']}', "
                f"cannot unnormalise with method '{method}'. Pass `method=None` or"
                f"`method='{self.config[var_ID]['method']}'`"
            )
        if method is None and unnorm:
            # Determine normalisation method from config for unnormalising
            method = self.config[var_ID]["method"]
        elif method not in self.valid_methods:
            raise ValueError(
                f"Method {method} not recognised. Use one of {self.valid_methods}"
            )

        params = self.get_config(var_ID, data, method)

        if method == "mean_std":
            std = params["std"]
            mean = params["mean"]
            if unnorm:
                scale = std
                offset = mean
            else:
                scale = 1 / std
                offset = -mean / std
            data = data * scale
            if add_offset:
                data = data + offset
            return data

        elif method == "min_max":
            minimum = params["min"]
            maximum = params["max"]
            if unnorm:
                scale = (maximum - minimum) / 2
                offset = (maximum + minimum) / 2
            else:
                scale = 2 / (maximum - minimum)
                offset = -(maximum + minimum) / (maximum - minimum)
            data = data * scale
            if add_offset:
                data = data + offset
            return data

    def map(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series],
        method: Optional[str] = None,
        add_offset: bool = True,
        unnorm: bool = False,
        assert_computed: bool = False,
    ):
        """Normalise or unnormalise the data values and coords in an xarray or
        pandas object.

        Args:
            data (:class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`pandas.DataFrame`, or :class:`pandas.Series`):
                [Description Needed]
            method (str, optional):
                [Description Needed]. Defaults to None.
            add_offset (bool, optional):
                [Description Needed]. Defaults to True.
            unnorm (bool, optional):
                [Description Needed]. Defaults to False.

        Returns:
            [Type]:
                [Description Needed]
        """
        if self.deepcopy:
            data = deepcopy(data)

        if isinstance(data, (xr.DataArray, xr.Dataset)) and not unnorm:
            self._validate_xr(data)
        elif isinstance(data, (pd.DataFrame, pd.Series)) and not unnorm:
            self._validate_pandas(data)

        if isinstance(data, (xr.DataArray, pd.Series)):
            # Single var
            var_ID = data.name
            if assert_computed:
                assert self.check_params_computed(
                    var_ID, method
                ), f"{method} normalisation params for {var_ID} not computed."
            data = self.map_array(data, var_ID, method, unnorm, add_offset)
        elif isinstance(data, (xr.Dataset, pd.DataFrame)):
            # Multiple vars
            for var_ID in data:
                if assert_computed:
                    assert self.check_params_computed(
                        var_ID, method
                    ), f"{method} normalisation params for {var_ID} not computed."
                data[var_ID] = self.map_array(
                    data[var_ID], var_ID, method, unnorm, add_offset
                )

        data = self.map_coords(data, unnorm=unnorm)

        return data

    def __call__(
        self,
        data: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
        ],
        method: str = "mean_std",
        assert_computed: bool = False,
    ) -> Union[
        xr.DataArray,
        xr.Dataset,
        pd.DataFrame,
        List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
    ]:
        """Normalise data.

        Args:
            data (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame` | List[:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame`]):
                Data to be normalised. Can be an xarray DataArray, xarray
                Dataset, pandas DataFrame, or a list containing objects of
                these types.
            method (str, optional): Normalisation method. Options include:
                - "mean_std": Normalise to mean=0 and std=1 (default)
                - "min_max": Normalise to min=-1 and max=1

        Returns:
            :class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame` | List[:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame`]:
                Normalised data. Type or structure depends on the input.
        """
        if isinstance(data, list):
            return [
                self.map(d, method, unnorm=False, assert_computed=assert_computed)
                for d in data
            ]
        else:
            return self.map(data, method, unnorm=False, assert_computed=assert_computed)

    def unnormalise(
        self,
        data: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
        ],
        add_offset: bool = True,
    ) -> Union[
        xr.DataArray,
        xr.Dataset,
        pd.DataFrame,
        List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
    ]:
        """Unnormalise data.

        Args:
            data (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame` | List[:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame`]):
                Data to unnormalise.
            add_offset (bool, optional):
                Whether to add the offset to the data when unnormalising. Set
                to False to unnormalise uncertainty values (e.g. std dev).
                Defaults to True.

        Returns:
            :class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame` | List[:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame`]:
                Unnormalised data.
        """
        if isinstance(data, list):
            return [self.map(d, add_offset=add_offset, unnorm=True) for d in data]
        else:
            return self.map(data, add_offset=add_offset, unnorm=True)


def xarray_to_coord_array_normalised(da: Union[xr.Dataset, xr.DataArray]) -> np.ndarray:
    """Convert xarray to normalised coordinate array.

    Args:
        da (:class:`xarray.Dataset` | :class:`xarray.DataArray`)
            ...

    Returns:
        :class:`numpy:numpy.ndarray`
            A normalised coordinate array of shape ``(2, N)``.
    """
    x1, x2 = da["x1"].values, da["x2"].values
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    return np.stack([X1.ravel(), X2.ravel()], axis=0)


def process_X_mask_for_X(X_mask: xr.DataArray, X: xr.DataArray) -> xr.DataArray:
    """Process X_mask by interpolating to X and converting to boolean.

    Both X_mask and X are xarray DataArrays with the same spatial coords.

    Args:
        X_mask (:class:`xarray.DataArray`):
            ...
        X (:class:`xarray.DataArray`):
            ...

    Returns:
        :class:`xarray.DataArray`
            ...
    """
    X_mask = X_mask.astype(float).interp_like(
        X, method="nearest", kwargs={"fill_value": 0}
    )
    X_mask.data = X_mask.data.astype(bool)
    X_mask.load()
    return X_mask


def mask_coord_array_normalised(
    coord_arr: np.ndarray, mask_da: Union[xr.DataArray, xr.Dataset, None]
) -> np.ndarray:
    """Remove points from (2, N) numpy array that are outside gridded xarray
    boolean mask.

    If `coord_arr` is shape `(2, N)`, then `mask_da` is a shape `(N,)` boolean
    array (True if point is inside mask, False if outside).

    Args:
        coord_arr (:class:`numpy:numpy.ndarray`):
            ...
        mask_da (:class:`xarray.Dataset` | :class:`xarray.DataArray`):
            ...

    Returns:
        :class:`numpy:numpy.ndarray`
            ...
    """
    if mask_da is None:
        return coord_arr
    mask_da = mask_da.astype(bool)
    x1, x2 = xr.DataArray(coord_arr[0]), xr.DataArray(coord_arr[1])
    mask_da = mask_da.sel(x1=x1, x2=x2, method="nearest")
    return coord_arr[:, mask_da]


def da1_da2_same_grid(da1: xr.DataArray, da2: xr.DataArray) -> bool:
    """Check if ``da1`` and ``da2`` are on the same grid.

    .. note::
        ``da1`` and ``da2`` are assumed normalised by ``DataProcessor``.

    Args:
        da1 (:class:`xarray.DataArray`):
            ...
        da2 (:class:`xarray.DataArray`):
            ...

    Returns:
        bool
            Whether ``da1`` and ``da2`` are on the same grid.
    """
    x1equal = np.array_equal(da1["x1"].values, da2["x1"].values)
    x2equal = np.array_equal(da1["x2"].values, da2["x2"].values)
    return x1equal and x2equal


def interp_da1_to_da2(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray:
    """Interpolate ``da1`` to ``da2``.

    .. note::
        ``da1`` and ``da2`` are assumed normalised by ``DataProcessor``.

    Args:
        da1 (:class:`xarray.DataArray`):
            ...
        da2 (:class:`xarray.DataArray`):
            ...

    Returns:
        :class:`xarray.DataArray`
            Interpolated xarray.
    """
    return da1.interp(x1=da2["x1"], x2=da2["x2"], method="nearest")
