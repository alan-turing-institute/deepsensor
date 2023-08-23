import numpy as np

import xarray as xr
import pandas as pd

import pprint

from copy import deepcopy

from typing import Union


class DataProcessor:
    """Normalise xarray and pandas data for use in deepsensor models"""

    def __init__(
        self,
        norm_params: dict = None,
        time_name: str = "time",
        x1_name: str = "x1",
        x2_name: str = "x2",
        x1_map: tuple = (0, 1),
        x2_map: tuple = (0, 1),
        deepcopy: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialise DataProcessor

        Args:
            norm_params (dict, optional): Normalisation params. Defaults to {}.
            x1_name (str, optional): Name of first spatial coord (e.g. "lat"). Defaults to "x1".
            x2_name (str, optional): Name of second spatial coord (e.g. "lon"). Defaults to "x2".
            x1_map (tuple, optional): 2-tuple of raw x1 coords to linearly map to (0, 1), respectively.
                Defaults to (0, 1) (i.e. no normalisation).
            x2_map (tuple, optional): 2-tuple of raw x2 coords to linearly map to (0, 1), respectively.
                Defaults to (0, 1) (i.e. no normalisation).
            deepcopy (bool, optional): Whether to make a deepcopy of raw data to ensure it is
                not changed by reference when normalising. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        x1_map, x2_map = self._validate_coord_mappings(x1_map, x2_map)

        if norm_params is None:
            norm_params = {}

        self.norm_params = norm_params

        if "coords" not in self.norm_params:
            # Add coordinate normalisation info to norm_params
            self.set_coord_params(time_name, x1_name, x1_map, x2_name, x2_map)

        self.raw_spatial_coord_names = [
            self.norm_params["coords"][coord]["name"] for coord in ["x1", "x2"]
        ]

        self.deepcopy = deepcopy
        self.verbose = verbose

        # List of valid normalisation method names
        self.valid_methods = ["mean_std", "min_max"]

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

        return x1_map, x2_map

    def _validate_xr(self, data: Union[xr.DataArray, xr.Dataset]):
        def _validate_da(da: xr.DataArray):
            coord_names = [
                self.norm_params["coords"][coord]["name"]
                for coord in ["time", "x1", "x2"]
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
            self.norm_params["coords"][coord]["name"] for coord in ["time", "x1", "x2"]
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
        s += pprint.pformat(self.norm_params)
        return s

    @classmethod
    def load_dask(cls, data):
        """Load dask data into memory"""
        if isinstance(data, xr.DataArray):
            data.load()
        elif isinstance(data, xr.Dataset):
            data.load()
        return data

    def set_coord_params(self, time_name, x1_name, x1_map, x2_name, x2_map):
        """Set coordinate normalisation params"""
        self.norm_params["coords"] = {}
        self.norm_params["coords"]["time"] = {"name": time_name}
        self.norm_params["coords"]["x1"] = {}
        self.norm_params["coords"]["x2"] = {}
        self.norm_params["coords"]["x1"]["name"] = x1_name
        self.norm_params["coords"]["x1"]["map"] = x1_map
        self.norm_params["coords"]["x2"]["name"] = x2_name
        self.norm_params["coords"]["x2"]["map"] = x2_map

    def check_params_computed(self, var_ID, method):
        """Check if normalisation params computed for a given variable"""
        if (
            var_ID in self.norm_params
            and self.norm_params[var_ID]["method"] == method
            and "param1" in self.norm_params[var_ID]
            and "param2" in self.norm_params[var_ID]
        ):
            return True
        else:
            return False

    def add_to_norm_params(self, var_ID, **kwargs):
        """Add `kwargs` to `norm_params` dict for variable `var_ID`"""
        self.norm_params[var_ID] = kwargs

    def get_norm_params(self, var_ID, data, method=None):
        """Get pre-computed normalisation params or compute them for variable `var_ID`"""
        if method not in self.valid_methods:
            raise ValueError(
                f"Method {method} not recognised. Must be one of {self.valid_methods}"
            )

        if self.check_params_computed(var_ID, method):
            # Already have "param1" and "param2" in norm_params with `"method": method` - load them
            param1 = self.norm_params[var_ID]["param1"]
            param2 = self.norm_params[var_ID]["param2"]
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
                param1 = float(data.mean())
                param2 = float(data.std())
            elif method == "min_max":
                param1 = float(data.min())
                param2 = float(data.max())
            if self.verbose:
                print(
                    f"Done. {var_ID} {method} param1={param1:.3f}, param2={param2:.3f}"
                )
            self.add_to_norm_params(
                var_ID, **{"method": method, "param1": param1, "param2": param2}
            )
        return param1, param2

    def map_coord_array(self, coord_array: np.ndarray, unnorm: bool = False):
        """Normalise or unnormalise a coordinate array

        Args:
            coord_array (np.ndarray): Array of shape (2, N) containing coords
            unnorm (bool, optional): Whether to unnormalise. Defaults to False.
        """
        x1, x2 = self.map_x1_and_x2(coord_array[0], coord_array[1], unnorm=unnorm)
        new_coords = np.stack([x1, x2], axis=0)
        return new_coords

    def map_x1_and_x2(self, x1: np.ndarray, x2: np.ndarray, unnorm: bool = False):
        """Normalise or unnormalise spatial coords in a array

        Args:
            x1 (np.ndarray): Array of shape (N_x1,) containing spatial coords of x1
            unnorm (bool, optional): Whether to unnormalise. Defaults to False.
        """
        x11, x12 = self.norm_params["coords"]["x1"]["map"]
        x21, x22 = self.norm_params["coords"]["x2"]["map"]

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
        """Normalise spatial coords in a pandas or xarray object"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Reset index to get coords as columns
            indexes = list(data.index.names)
            data = data.reset_index()

        if unnorm:
            new_coord_IDs = [
                self.norm_params["coords"][coord_ID]["name"]
                for coord_ID in ["time", "x1", "x2"]
            ]
            old_coord_IDs = ["time", "x1", "x2"]
        else:
            new_coord_IDs = ["time", "x1", "x2"]
            old_coord_IDs = [
                self.norm_params["coords"][coord_ID]["name"]
                for coord_ID in ["time", "x1", "x2"]
            ]

        new_x1, new_x2 = self.map_x1_and_x2(
            data[old_coord_IDs[1]], data[old_coord_IDs[2]], unnorm=unnorm
        )

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
        method: str = None,
        unnorm: bool = False,
        add_offset=True,
    ):
        """Normalise or unnormalise the data values in an xarray, pandas, or numpy object"""
        if not unnorm and method is None:
            raise ValueError("Must provide `method` if normalising data.")
        elif (
            unnorm
            and method is not None
            and self.norm_params[var_ID]["method"] != method
        ):
            # User has provided a different method to the one used for normalising
            raise ValueError(
                f"Variable '{var_ID}' has been normalised with method '{self.norm_params[var_ID]['method']}', "
                f"cannot unnormalise with method '{method}'. Pass `method=None` or"
                f"`method='{self.norm_params[var_ID]['method']}'`"
            )
        if method is None and unnorm:
            # Determine normalisation method from norm_params for unnormalising
            method = self.norm_params[var_ID]["method"]
        elif method not in self.valid_methods:
            raise ValueError(
                f"Method {method} not recognised. Use one of {self.valid_methods}"
            )

        param1, param2 = self.get_norm_params(var_ID, data, method)

        if method == "mean_std":
            if unnorm:
                scale = param2
                offset = param1
            else:
                scale = 1 / param2
                offset = -param1 / param2
            data = data * scale
            if add_offset:
                data = data + offset
            return data

        elif method == "min_max":
            if unnorm:
                scale = (param2 - param1) / 2
                offset = (param2 + param1) / 2
            else:
                scale = 2 / (param2 - param1)
                offset = -(param2 + param1) / (param2 - param1)
            data = data * scale
            if add_offset:
                data = data + offset
            return data

    def map(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series],
        method: str = None,
        add_offset: bool = True,
        unnorm: bool = False,
    ):
        """Normalise or unnormalise the data values and coords in an xarray or pandas object"""
        if self.deepcopy:
            data = deepcopy(data)

        if isinstance(data, (xr.DataArray, xr.Dataset)) and not unnorm:
            self._validate_xr(data)
        elif isinstance(data, (pd.DataFrame, pd.Series)) and not unnorm:
            self._validate_pandas(data)

        if isinstance(data, (xr.DataArray, pd.Series)):
            # Single var
            data = self.map_array(data, data.name, method, unnorm, add_offset)
        elif isinstance(data, (xr.Dataset, pd.DataFrame)):
            # Multiple vars
            for var_ID in data:
                data[var_ID] = self.map_array(
                    data[var_ID], var_ID, method, unnorm, add_offset
                )

        data = self.map_coords(data, unnorm=unnorm)

        return data

    def __call__(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, list],
        method: str = "mean_std",
    ) -> Union[xr.DataArray, xr.Dataset, pd.DataFrame, list]:
        """Normalise data

        Args:
            data (Union[xr.DataArray, xr.Dataset, pd.DataFrame, list]): Data to normalise
            method (str, optional): Normalisation method. Defaults to "mean_std". Options:
                - "mean_std": Normalise to mean=0 and std=1
                - "min_max": Normalise to min=-1 and max=1
        """
        if isinstance(data, list):
            return [self.map(d, method, unnorm=False) for d in data]
        else:
            return self.map(data, method, unnorm=False)

    def unnormalise(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, list],
        add_offset: bool = True,
    ) -> Union[xr.DataArray, xr.Dataset, pd.DataFrame, list]:
        """Unnormalise data

        Args:
            data (Union[xr.DataArray, xr.Dataset, pd.DataFrame, list]): Data to unnormalise
            add_offset (bool, optional): Whether to add the offset to the data when unnormalising.
                Set to False to unnormalise uncertainty values (e.g. std dev). Defaults to True.
        """
        if isinstance(data, list):
            return [self.map(d, add_offset=add_offset, unnorm=True) for d in data]
        else:
            return self.map(data, add_offset=add_offset, unnorm=True)


def xarray_to_coord_array_normalised(da: Union[xr.Dataset, xr.DataArray]):
    x1, x2 = da["x1"].values, da["x2"].values
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    return np.stack([X1.ravel(), X2.ravel()], axis=0)


def mask_coord_array_normalised(coord_arr, mask_da):
    if mask_da is None:
        return coord_arr
    mask_da = mask_da.astype(float)  # Temporarily convert to float for interpolation
    mask_da = mask_da.interp(
        {"x1": xr.DataArray(coord_arr[0]), "x2": xr.DataArray(coord_arr[1])},
        method="nearest",
        kwargs=dict(fill_value=None, bounds_error=False),
    ).data.astype(
        bool
    )  # Shape `coord_arr.shape[1]`, False if point is outside mask
    return coord_arr[:, mask_da]


def da1_da2_same_grid(da1: xr.DataArray, da2: xr.DataArray) -> bool:
    """Check if da1 and da2 are on the same grid

    Note: da1 and da2 are assumed normalised by DataProcessor.
    """
    x1equal = np.array_equal(da1["x1"].values, da2["x1"].values)
    x2equal = np.array_equal(da1["x2"].values, da2["x2"].values)
    return x1equal and x2equal


def interp_da1_to_da2(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray:
    """Interpolate da1 to da2

    Note: da1 and da2 are assumed normalised by DataProcessor.
    """
    return da1.interp(x1=da2["x1"], x2=da2["x2"], method="nearest")
