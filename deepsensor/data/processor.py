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
        if norm_params is None:
            norm_params = {}

        self.norm_params = norm_params

        if "coords" not in self.norm_params:
            # Add coordinate normalisation info to norm_params
            self.set_coord_params(x1_name, x1_map, x2_name, x2_map)

        self.deepcopy = deepcopy
        self.verbose = verbose

        self.valid_methods = ["mean_std", "min_max"]

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
    
    def set_coord_params(self, x1_name, x1_map, x2_name, x2_map):
        """Set coordinate normalisation params"""
        x1_map = np.array(x1_map)
        x2_map = np.array(x2_map)
        self.norm_params["coords"] = {}
        self.norm_params["coords"]["x1"] = {}
        self.norm_params["coords"]["x2"] = {}
        self.norm_params["coords"]["x1"]["name"] = x1_name
        self.norm_params["coords"]["x1"]["map"] = x1_map
        self.norm_params["coords"]["x2"]["name"] = x2_name
        self.norm_params["coords"]["x2"]["map"] = x2_map

        # Check that map is not two of the same number
        if np.diff(x1_map) == 0:
            raise ValueError(f"x1_map must be a 2-tuple of different numbers, not {x1_map}")
        if np.diff(x2_map) == 0:
            raise ValueError(f"x2_map must be a 2-tuple of different numbers, not {x2_map}")

    def check_params_computed(self, var_ID, param1_ID, param2_ID):
        """Check if normalisation params computed for a given variable"""
        if (
            var_ID in self.norm_params
            and param1_ID in self.norm_params[var_ID]
            and param2_ID in self.norm_params[var_ID]
        ):
            return True
        else:
            return False

    def add_to_norm_params(self, var_ID, **kwargs):
        """Add to `norm_params` dict for variable `var_ID`"""
        self.norm_params[var_ID] = kwargs

    def get_norm_params(self, var_ID, data, method="mean_std", unnorm=False):
        """Get pre-computed normalisation params or compute them for variable `var_ID`

        If `unnorm` is True, raises a ValueError if normalisation params have not been computed.
        """
        if method == "mean_std":
            param1_ID, param2_ID = "mean", "std"
        elif method == "min_max":
            param1_ID, param2_ID = "min", "max"
        else:
            raise ValueError(
                f"Method {method} not recognised. Must be one of {self.valid_methods}"
            )

        if not self.check_params_computed(var_ID, param1_ID, param2_ID):
            if unnorm:
                raise ValueError(
                    f"Normalisation params for {var_ID} not computed. Cannot unnormalise."
                )
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
                    f"Done. {var_ID} {param1_ID}={param1:.3f}, {param2_ID}={param2:.3f}"
                )
            self.add_to_norm_params(var_ID, **{param1_ID: param1, param2_ID: param2})
        else:
            param1 = self.norm_params[var_ID][param1_ID]
            param2 = self.norm_params[var_ID][param2_ID]
        return param1, param2

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

    def map_coords(self, data, unnorm=False):
        """Normalise spatial coords in a pandas or xarray object"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Reset index to get coords as columns
            indexes = list(data.index.names)
            data = data.reset_index()

        if not unnorm:
            new_coord_IDs = ["x1", "x2"]
            old_coord_IDs = [
                self.norm_params["coords"][coord_ID]["name"]
                for coord_ID in ["x1", "x2"]
            ]
        else:
            new_coord_IDs = [
                self.norm_params["coords"][coord_ID]["name"]
                for coord_ID in ["x1", "x2"]
            ]
            old_coord_IDs = ["x1", "x2"]

        new_coord_tensor = self.map_x1_and_x2(
            data[old_coord_IDs[0]], data[old_coord_IDs[1]], unnorm=unnorm
        )
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Add coords to dataframe
            data[new_coord_IDs[0]] = new_coord_tensor[0]
            data[new_coord_IDs[1]] = new_coord_tensor[1]
            # Drop old coord columns
            data = data.drop(columns=old_coord_IDs)
        elif isinstance(data, (xr.DataArray, xr.Dataset)):
            data = data.assign_coords(
                {
                    old_coord_IDs[0]: new_coord_tensor[0],
                    old_coord_IDs[1]: new_coord_tensor[1],
                }
            )
            data = data.rename(
                {old_coord_IDs[0]: new_coord_IDs[0], old_coord_IDs[1]: new_coord_IDs[1]}
            )

        if isinstance(data, (pd.DataFrame, pd.Series)):
            # Set index back to original
            [indexes.remove(old_coord_ID) for old_coord_ID in old_coord_IDs]
            indexes.extend(new_coord_IDs)
            data = data.set_index(indexes)
        return data

    def map(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series],
        method: str = "mean_std",
        add_offset: bool = True,
        unnorm: bool = False,
    ):
        """Normalise or unnormalise data"""
        if self.deepcopy:
            data = deepcopy(data)

        def mapper(data, param1, param2, method):
            if method == "mean_std":
                if not unnorm:
                    scale = 1 / param2
                    offset = -param1 / param2
                else:
                    scale = param2
                    offset = param1
            elif method == "min_max":
                if not unnorm:
                    scale = 2 / (param2 - param1)
                    offset = -(param2 + param1) / (param2 - param1)
                else:
                    scale = (param2 - param1) / 2
                    offset = (param2 + param1) / 2
            data = data * scale
            if add_offset:
                data = data + offset
            return data

        if isinstance(data, (xr.DataArray, pd.Series)):
            # Single var
            var_ID = data.name
            param1, param2 = self.get_norm_params(var_ID, data, method, unnorm)
            data = mapper(data, param1, param2, method)
        elif isinstance(data, (xr.Dataset, pd.DataFrame)):
            # Multiple vars
            for var_ID in data:
                param1, param2 = self.get_norm_params(
                    var_ID, data[var_ID], method, unnorm
                )
                data[var_ID] = mapper(data[var_ID], param1, param2, method)

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
            method (str, optional): Normalisation method. Options:
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
        method: str = "mean_std",
    ) -> Union[xr.DataArray, xr.Dataset, pd.DataFrame, list]:
        """Unnormalise data

        Args:
            data (Union[xr.DataArray, xr.Dataset, pd.DataFrame, list]): Data to unnormalise
            method (str, optional): Normalisation method. Options:
                - "mean_std": Normalise to mean=0 and std=1
                - "min_max": Normalise to min=-1 and max=1
            add_offset (bool, optional): Whether to add the offset to the data when unnormalising.
                Set to False to unnormalise uncertainty values (e.g. std dev). Defaults to True.
        """
        if isinstance(data, list):
            return [self.map(d, method, add_offset, unnorm=True) for d in data]
        else:
            return self.map(data, method, add_offset, unnorm=True)
