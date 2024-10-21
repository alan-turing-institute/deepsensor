from deepsensor.data.task import Task, flatten_X

import os
import json
import copy

import numpy as np
import xarray as xr
import pandas as pd

from typing import List, Tuple, Union, Optional

from deepsensor.errors import InvalidSamplingStrategyError


class TaskLoader:
    """Generates :class:`~.data.task.Task` objects for training, testing, and inference with DeepSensor models.

    Provides a suite of sampling methods for generating :class:`~.data.task.Task` objects for different kinds of
    predictions, such as: spatial interpolation, forecasting, downscaling, or some combination
    of these.

    The behaviour is the following:
        - If all data passed as paths, load the data and overwrite the paths with the loaded data
        - Either all data is passed as paths, or all data is passed as loaded data (else ``ValueError``)
        - If all data passed as paths, the TaskLoader can be saved with the ``save`` method
          (using config)

    Args:
        task_loader_ID:
            If loading a TaskLoader from a config file, this is the folder the
            TaskLoader was saved in (using `.save`). If this argument is passed, all other
            arguments are ignored.
        context (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame` | List[:class:`xarray.DataArray` | :class:`xarray.Dataset`, :class:`pandas.DataFrame`]):
            Context data. Can be a single :class:`xarray.DataArray`,
            :class:`xarray.Dataset` or :class:`pandas.DataFrame`, or a
            list/tuple of these.
        target (:class:`xarray.DataArray` | :class:`xarray.Dataset` | :class:`pandas.DataFrame` | List[:class:`xarray.DataArray` | :class:`xarray.Dataset`, :class:`pandas.DataFrame`]):
            Target data. Can be a single :class:`xarray.DataArray`,
            :class:`xarray.Dataset` or :class:`pandas.DataFrame`, or a
            list/tuple of these.
        aux_at_contexts (Tuple[int, :class:`xarray.DataArray` | :class:`xarray.Dataset`], optional):
            Auxiliary data at context locations. Tuple of two elements, where
            the first element is the index of the context set for which the
            auxiliary data will be sampled at, and the second element is the
            auxiliary data, which can be a single :class:`xarray.DataArray` or
            :class:`xarray.Dataset`. Default: None.
        aux_at_targets (:class:`xarray.DataArray` | :class:`xarray.Dataset`, optional):
            Auxiliary data at target locations. Can be a single
            :class:`xarray.DataArray` or :class:`xarray.Dataset`. Default:
            None.
        links (Tuple[int, int] | List[Tuple[int, int]], optional):
            Specifies links between context and target data. Each link is a
            tuple of two integers, where the first integer is the index of the
            context data and the second integer is the index of the target
            data. Can be a single tuple in the case of a single link. If None,
            no links are specified. Default: None.
        context_delta_t (int | List[int], optional):
            Time difference between context data and t=0 (task init time). Can
            be a single int (same for all context data) or a list/tuple of
            ints. Default is 0.
        target_delta_t (int | List[int], optional):
            Time difference between target data and t=0 (task init time). Can
            be a single int (same for all target data) or a list/tuple of ints.
            Default is 0.
        time_freq (str, optional):
            Time frequency of the data. Default: ``'D'`` (daily).
        xarray_interp_method (str, optional):
            Interpolation method to use when interpolating
            :class:`xarray.DataArray`. Default is ``'linear'``.
        discrete_xarray_sampling (bool, optional):
            When randomly sampling xarray variables, whether to sample at
            discrete points defined at grid cell centres, or at continuous
            points within the grid. Default is ``False``.
        dtype (object, optional):
            Data type of the data. Used to cast the data to the specified
            dtype. Default: ``'float32'``.
    """

    config_fname = "task_loader_config.json"

    def __init__(
        self,
        task_loader_ID: Union[str, None] = None,
        context: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            str,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame, str]],
        ] = None,
        target: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            str,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame, str]],
        ] = None,
        aux_at_contexts: Optional[Tuple[int, Union[xr.DataArray, xr.Dataset]]] = None,
        aux_at_targets: Optional[
            Union[
                xr.DataArray,
                xr.Dataset,
            ]
        ] = None,
        links: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
        context_delta_t: Union[int, List[int]] = 0,
        target_delta_t: Union[int, List[int]] = 0,
        time_freq: str = "D",
        xarray_interp_method: str = "linear",
        discrete_xarray_sampling: bool = False,
        dtype: object = "float32",
    ) -> None:
        if task_loader_ID is not None:
            self.task_loader_ID = task_loader_ID
            # Load TaskLoader from config file
            fpath = os.path.join(task_loader_ID, self.config_fname)
            with open(fpath, "r") as f:
                self.config = json.load(f)

            self.context = self.config["context"]
            self.target = self.config["target"]
            self.aux_at_contexts = self.config["aux_at_contexts"]
            self.aux_at_targets = self.config["aux_at_targets"]
            self.links = self.config["links"]
            if self.links is not None:
                self.links = [tuple(link) for link in self.links]
            self.context_delta_t = self.config["context_delta_t"]
            self.target_delta_t = self.config["target_delta_t"]
            self.time_freq = self.config["time_freq"]
            self.xarray_interp_method = self.config["xarray_interp_method"]
            self.discrete_xarray_sampling = self.config["discrete_xarray_sampling"]
            self.dtype = self.config["dtype"]
        else:
            self.context = context
            self.target = target
            self.aux_at_contexts = aux_at_contexts
            self.aux_at_targets = aux_at_targets
            self.links = links
            self.context_delta_t = context_delta_t
            self.target_delta_t = target_delta_t
            self.time_freq = time_freq
            self.xarray_interp_method = xarray_interp_method
            self.discrete_xarray_sampling = discrete_xarray_sampling
            self.dtype = dtype

        if not isinstance(self.context, (tuple, list)):
            self.context = (self.context,)
        if not isinstance(self.target, (tuple, list)):
            self.target = (self.target,)

        if isinstance(self.context_delta_t, int):
            self.context_delta_t = (self.context_delta_t,) * len(self.context)
        else:
            assert len(self.context_delta_t) == len(self.context), (
                f"Length of context_delta_t ({len(self.context_delta_t)}) must be the same as "
                f"the number of context sets ({len(self.context)})"
            )
        if isinstance(self.target_delta_t, int):
            self.target_delta_t = (self.target_delta_t,) * len(self.target)
        else:
            assert len(self.target_delta_t) == len(self.target), (
                f"Length of target_delta_t ({len(self.target_delta_t)}) must be the same as "
                f"the number of target sets ({len(self.target)})"
            )

        all_paths = self._check_if_all_data_passed_as_paths()
        if all_paths:
            self._set_config()
            self._load_data_from_paths()

        self.context = self._cast_to_dtype(self.context)
        self.target = self._cast_to_dtype(self.target)
        self.aux_at_contexts = self._cast_to_dtype(self.aux_at_contexts)
        self.aux_at_targets = self._cast_to_dtype(self.aux_at_targets)

        self.links = self._check_links(self.links)

        (
            self.context_dims,
            self.target_dims,
            self.aux_at_target_dims,
        ) = self.count_context_and_target_data_dims()
        (
            self.context_var_IDs,
            self.target_var_IDs,
            self.context_var_IDs_and_delta_t,
            self.target_var_IDs_and_delta_t,
            self.aux_at_target_var_IDs,
        ) = self.infer_context_and_target_var_IDs()

    def _set_config(self):
        """Instantiate a config dictionary for the TaskLoader object."""
        # Take deepcopy to avoid modifying the original config
        self.config = copy.deepcopy(
            dict(
                context=self.context,
                target=self.target,
                aux_at_contexts=self.aux_at_contexts,
                aux_at_targets=self.aux_at_targets,
                links=self.links,
                context_delta_t=self.context_delta_t,
                target_delta_t=self.target_delta_t,
                time_freq=self.time_freq,
                xarray_interp_method=self.xarray_interp_method,
                discrete_xarray_sampling=self.discrete_xarray_sampling,
                dtype=self.dtype,
            )
        )

    def _check_if_all_data_passed_as_paths(self) -> bool:
        """If all data passed as paths, save paths to config and return True."""

        def _check_if_strings(x, mode="all"):
            if x is None:
                return None
            elif isinstance(x, (tuple, list)):
                if mode == "all":
                    return all([isinstance(x_i, str) for x_i in x])
                elif mode == "any":
                    return any([isinstance(x_i, str) for x_i in x])
            else:
                return isinstance(x, str)

        all_paths = all(
            filter(
                lambda x: x is not None,
                [
                    _check_if_strings(self.context),
                    _check_if_strings(self.target),
                    _check_if_strings(self.aux_at_contexts),
                    _check_if_strings(self.aux_at_targets),
                ],
            )
        )
        self._is_saveable = all_paths

        any_paths = any(
            filter(
                lambda x: x is not None,
                [
                    _check_if_strings(self.context, mode="any"),
                    _check_if_strings(self.target, mode="any"),
                    _check_if_strings(self.aux_at_contexts, mode="any"),
                    _check_if_strings(self.aux_at_targets, mode="any"),
                ],
            )
        )
        if any_paths and not all_paths:
            raise ValueError(
                "Data must be passed either all as paths or all as xarray/pandas objects (not a mix)."
            )

        return all_paths

    def _load_data_from_paths(self):
        """Load data from paths and overwrite paths with loaded data."""
        loaded_data = {}

        def _load_pandas_or_xarray(path):
            # Need to be careful about this. We need to ensure data gets into the right form
            #  for TaskLoader.
            if path is None:
                return None
            elif path in loaded_data:
                return loaded_data[path]
            elif path.endswith(".nc"):
                data = xr.open_dataset(path)
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index(["time", "x1", "x2"]).sort_index()
                else:
                    df = df.set_index(["x1", "x2"]).sort_index()
                data = df
            else:
                raise ValueError(f"Unknown file extension for {path}")
            loaded_data[path] = data
            return data

        def _load_data(data):
            if isinstance(data, (tuple, list)):
                data = tuple([_load_pandas_or_xarray(data_i) for data_i in data])
            else:
                data = _load_pandas_or_xarray(data)
            return data

        self.context = _load_data(self.context)
        self.target = _load_data(self.target)
        self.aux_at_contexts = _load_data(self.aux_at_contexts)
        self.aux_at_targets = _load_data(self.aux_at_targets)

    def save(self, folder: str):
        """Save TaskLoader config to JSON in `folder`."""
        if not self._is_saveable:
            raise ValueError(
                "TaskLoader cannot be saved because not all data was passed as paths."
            )

        os.makedirs(folder, exist_ok=True)
        fpath = os.path.join(folder, self.config_fname)
        with open(fpath, "w") as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

    def _cast_to_dtype(
        self,
        var: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame, str]],
        ],
    ) -> (List, List):
        """Cast context and target data to the default dtype.

        ..
            TODO unit test this by passing in a variety of data types and
            checking that they are cast correctly.

        Args:
            var : ...
                ...

        Returns:
            tuple: Tuple of context data with specified dtype.
            tuple: Tuple of target data with specified dtype.
        """

        def cast_to_dtype(var):
            if isinstance(var, xr.DataArray):
                var = var.astype(self.dtype)
                var["x1"] = var["x1"].astype(self.dtype)
                var["x2"] = var["x2"].astype(self.dtype)
            elif isinstance(var, xr.Dataset):
                var = var.astype(self.dtype)
                var["x1"] = var["x1"].astype(self.dtype)
                var["x2"] = var["x2"].astype(self.dtype)
            elif isinstance(var, (pd.DataFrame, pd.Series)):
                var = var.astype(self.dtype)
                # Note: Numeric pandas indexes are always cast to float64, so we have to cast
                #   x1/x2 coord dtypes during task sampling
            else:
                raise ValueError(f"Unknown type {type(var)} for context set {var}")
            return var

        if var is None:
            return var
        elif isinstance(var, (tuple, list)):
            var = tuple([cast_to_dtype(var_i) for var_i in var])
        else:
            var = cast_to_dtype(var)

        return var

    def load_dask(self) -> None:
        """Load any `dask` data into memory.

        This function triggers the computation and loading of any data that
        is represented as dask arrays or datasets into memory.

        Returns:
            None
        """

        def load(datasets):
            if datasets is None:
                return
            if not isinstance(datasets, (tuple, list)):
                datasets = [datasets]
            for i, var in enumerate(datasets):
                if isinstance(var, (xr.DataArray, xr.Dataset)):
                    var = var.load()

        load(self.context)
        load(self.target)
        load(self.aux_at_contexts)
        load(self.aux_at_targets)

        return None

    def count_context_and_target_data_dims(self):
        """Count the number of data dimensions in the context and target data.

        Returns:
            tuple: context_dims, Tuple of data dimensions in the context data.
            tuple: target_dims, Tuple of data dimensions in the target data.

        Raises:
            ValueError: If the context/target data is not a tuple/list of
                        :class:`xarray.DataArray`, :class:`xarray.Dataset` or
                        :class:`pandas.DataFrame`.
        """

        def count_data_dims_of_tuple_of_sets(datasets):
            if not isinstance(datasets, (tuple, list)):
                datasets = [datasets]

            dims = []
            # Distinguish between xr.DataArray, xr.Dataset and pd.DataFrame
            for i, var in enumerate(datasets):
                if isinstance(var, xr.Dataset):
                    dim = len(var.data_vars)  # Multiple data variables
                elif isinstance(var, xr.DataArray):
                    dim = 1  # Single data variable
                elif isinstance(var, pd.DataFrame):
                    dim = len(var.columns)  # Assumes all columns are data variables
                elif isinstance(var, pd.Series):
                    dim = 1  # Single data variable
                else:
                    raise ValueError(f"Unknown type {type(var)} for context set {var}")
                dims.append(dim)
            return dims

        context_dims = count_data_dims_of_tuple_of_sets(self.context)
        target_dims = count_data_dims_of_tuple_of_sets(self.target)
        if self.aux_at_contexts is not None:
            context_dims += count_data_dims_of_tuple_of_sets(self.aux_at_contexts)
        if self.aux_at_targets is not None:
            aux_at_target_dims = count_data_dims_of_tuple_of_sets(self.aux_at_targets)[
                0
            ]
        else:
            aux_at_target_dims = 0

        return tuple(context_dims), tuple(target_dims), aux_at_target_dims

    def infer_context_and_target_var_IDs(self):
        """Infer the variable IDs of the context and target data.

        Returns:
            tuple: context_var_IDs, Tuple of variable IDs in the context data.
            tuple: target_var_IDs, Tuple of variable IDs in the target data.

        Raises:
            ValueError: If the context/target data is not a tuple/list of
                        :class:`xarray.DataArray`, :class:`xarray.Dataset` or
                        :class:`pandas.DataFrame`.
        """

        def infer_var_IDs_of_tuple_of_sets(datasets, delta_ts=None):
            """If delta_ts is not None, then add the delta_t to the variable ID."""
            if not isinstance(datasets, (tuple, list)):
                datasets = [datasets]

            var_IDs = []
            # Distinguish between xr.DataArray, xr.Dataset and pd.DataFrame
            for i, var in enumerate(datasets):
                if isinstance(var, xr.DataArray):
                    var_ID = (var.name,)  # Single data variable
                elif isinstance(var, xr.Dataset):
                    var_ID = tuple(var.data_vars.keys())  # Multiple data variables
                elif isinstance(var, pd.DataFrame):
                    var_ID = tuple(var.columns)
                elif isinstance(var, pd.Series):
                    var_ID = (var.name,)
                else:
                    raise ValueError(f"Unknown type {type(var)} for context set {var}")

                if delta_ts is not None:
                    # Add delta_t to the variable ID
                    var_ID = tuple(
                        [f"{var_ID_i}_t{delta_ts[i]}" for var_ID_i in var_ID]
                    )
                else:
                    var_ID = tuple([f"{var_ID_i}" for var_ID_i in var_ID])

                var_IDs.append(var_ID)

            return var_IDs

        context_var_IDs = infer_var_IDs_of_tuple_of_sets(self.context)
        context_var_IDs_and_delta_t = infer_var_IDs_of_tuple_of_sets(
            self.context, self.context_delta_t
        )
        target_var_IDs = infer_var_IDs_of_tuple_of_sets(self.target)
        target_var_IDs_and_delta_t = infer_var_IDs_of_tuple_of_sets(
            self.target, self.target_delta_t
        )

        if self.aux_at_contexts is not None:
            context_var_IDs += infer_var_IDs_of_tuple_of_sets(self.aux_at_contexts)
            context_var_IDs_and_delta_t += infer_var_IDs_of_tuple_of_sets(
                self.aux_at_contexts, [0]
            )

        if self.aux_at_targets is not None:
            aux_at_target_var_IDs = infer_var_IDs_of_tuple_of_sets(self.aux_at_targets)[
                0
            ]
        else:
            aux_at_target_var_IDs = None

        return (
            tuple(context_var_IDs),
            tuple(target_var_IDs),
            tuple(context_var_IDs_and_delta_t),
            tuple(target_var_IDs_and_delta_t),
            aux_at_target_var_IDs,
        )

    def _check_links(self, links: Union[Tuple[int, int], List[Tuple[int, int]]]):
        """Check that the context-target links are valid.

        Args:
            links (Tuple[int, int] | List[Tuple[int, int]]):
                Specifies links between context and target data. Each link is a
                tuple of two integers, where the first integer is the index of
                the context data and the second integer is the index of the
                target data. Can be a single tuple in the case of a single
                link. If None, no links are specified. Default: None.

        Returns:
            Tuple[int, int] | List[Tuple[int, int]]
                The input links, if valid.

        Raises:
            ValueError
                If the links are not valid.
        """
        if links is None:
            return None

        assert isinstance(
            links, list
        ), f"Links must be a list of length-2 tuples, but got {type(links)}"
        assert len(links) > 0, "If links is not None, it must be a non-empty list"
        assert all(
            isinstance(link, tuple) for link in links
        ), f"Links must be a list of tuples, but got {[type(link) for link in links]}"
        assert all(
            len(link) == 2 for link in links
        ), f"Links must be a list of length-2 tuples, but got lengths {[len(link) for link in links]}"

        # Check that the links are valid
        for link_i, (context_idx, target_idx) in enumerate(links):
            if context_idx >= len(self.context):
                raise ValueError(
                    f"Invalid context index {context_idx} in link {link_i} of {links}: "
                    f"there are only {len(self.context)} context sets"
                )
            if target_idx >= len(self.target):
                raise ValueError(
                    f"Invalid target index {target_idx} in link {link_i} of {links}: "
                    f"there are only {len(self.target)} target sets"
                )

        return links

    def __str__(self):
        """String representation of the TaskLoader object (user-friendly)."""
        s = f"TaskLoader({len(self.context_dims)} context sets, {len(self.target_dims)} target sets)"
        s += f"\nContext variable IDs: {self.context_var_IDs}"
        s += f"\nTarget variable IDs: {self.target_var_IDs}"
        if self.aux_at_targets is not None:
            s += f"\nAuxiliary-at-target variable IDs: {self.aux_at_target_var_IDs}"
        return s

    def __repr__(self):
        """Representation of the TaskLoader object (for developers).

        ..
            TODO make this a more verbose version of __str__
        """
        s = str(self)
        s += "\n"
        s += f"\nContext data dimensions: {self.context_dims}"
        s += f"\nTarget data dimensions: {self.target_dims}"
        if self.aux_at_targets is not None:
            s += f"\nAuxiliary-at-target data dimensions: {self.aux_at_target_dims}"
        return s

    def sample_da(
        self,
        da: Union[xr.DataArray, xr.Dataset],
        sampling_strat: Union[str, int, float, np.ndarray],
        seed: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray):
        """Sample a DataArray according to a given strategy.

        Args:
            da (:class:`xarray.DataArray` | :class:`xarray.Dataset`):
                DataArray to sample, assumed to be sliced for the task already.
            sampling_strat (str | int | float | :class:`numpy:numpy.ndarray`):
                Sampling strategy, either "all" or an integer for random grid
                cell sampling.
            seed (int, optional):
                Seed for random sampling. Default is None.

        Returns:
            Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]:
                Tuple of sampled target data and sampled context data.

        Raises:
            InvalidSamplingStrategyError:
                If the sampling strategy is not valid or if a numpy coordinate
                array is passed to sample an xarray object, but the coordinates
                are out of bounds.
        """
        da = da.load()  # Converts dask -> numpy if not already loaded
        if isinstance(da, xr.Dataset):
            da = da.to_array()

        if isinstance(sampling_strat, float):
            sampling_strat = int(sampling_strat * da.size)

        if isinstance(sampling_strat, (int, np.integer)):
            N = sampling_strat
            rng = np.random.default_rng(seed)
            if self.discrete_xarray_sampling:
                x1 = rng.choice(da.coords["x1"].values, N, replace=True)
                x2 = rng.choice(da.coords["x2"].values, N, replace=True)
                Y_c = da.sel(x1=xr.DataArray(x1), x2=xr.DataArray(x2)).data
            elif not self.discrete_xarray_sampling:
                if N == 0:
                    # Catch zero-context edge case before interp fails
                    X_c = np.zeros((2, 0), dtype=self.dtype)
                    dim = da.shape[0] if da.ndim == 3 else 1
                    Y_c = np.zeros((dim, 0), dtype=self.dtype)
                    return X_c, Y_c
                x1 = rng.uniform(da.coords["x1"].min(), da.coords["x1"].max(), N)
                x2 = rng.uniform(da.coords["x2"].min(), da.coords["x2"].max(), N)
                Y_c = da.sel(x1=xr.DataArray(x1), x2=xr.DataArray(x2), method="nearest")
                Y_c = np.array(Y_c, dtype=self.dtype)
            X_c = np.array([x1, x2], dtype=self.dtype)

        elif isinstance(sampling_strat, np.ndarray):
            X_c = sampling_strat.astype(self.dtype)
            try:
                Y_c = da.sel(
                    x1=xr.DataArray(X_c[0]),
                    x2=xr.DataArray(X_c[1]),
                    method="nearest",
                    tolerance=0.1,  # Maximum distance from observed point to sample
                )
            except KeyError:
                raise InvalidSamplingStrategyError(
                    f"Passed a numpy coordinate array to sample xarray object, "
                    f"but the coordinates are out of bounds."
                )
            Y_c = np.array(Y_c, dtype=self.dtype)

        elif sampling_strat in ["all", "gapfill"]:
            X_c = (
                da.coords["x1"].values[np.newaxis],
                da.coords["x2"].values[np.newaxis],
            )
            Y_c = da.data
            if Y_c.ndim == 2:
                # returned a 2D array, but we need a 3D array of shape (variable, N_x1, N_x2)
                Y_c = Y_c.reshape(1, *Y_c.shape)
        else:
            raise InvalidSamplingStrategyError(
                f"Unknown sampling strategy {sampling_strat}"
            )

        if Y_c.ndim == 1:
            # returned a 1D array, but we need a 2D array of shape (variable, N)
            Y_c = Y_c.reshape(1, *Y_c.shape)

        return X_c, Y_c

    def sample_df(
        self,
        df: Union[pd.DataFrame, pd.Series],
        sampling_strat: Union[str, int, float, np.ndarray],
        seed: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray):
        """Sample a DataFrame according to a given strategy.

        Args:
            df (:class:`pandas.DataFrame` | :class:`pandas.Series`):
                Dataframe to sample, assumed to be time-sliced for the task
                already.
            sampling_strat (str | int | float | :class:`numpy:numpy.ndarray`):
                Sampling strategy, either "all" or an integer for random grid
                cell sampling.
            seed (int, optional):
                Seed for random sampling. Default is None.

        Returns:
            Tuple[X_c, Y_c]:
                Tuple of sampled target data and sampled context data.

        Raises:
            InvalidSamplingStrategyError:
                If the sampling strategy is not valid or if a numpy coordinate
                array is passed to sample a pandas object, but the DataFrame
                does not contain all the requested samples.
        """
        df = df.dropna(how="any")  # If any obs are NaN, drop them

        if isinstance(sampling_strat, float):
            sampling_strat = int(sampling_strat * df.shape[0])

        if isinstance(sampling_strat, (int, np.integer)):
            N = sampling_strat
            rng = np.random.default_rng(seed)
            idx = rng.choice(df.index, N)
            X_c = df.loc[idx].reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_c = df.loc[idx].values.T
        elif isinstance(sampling_strat, str) and sampling_strat in [
            "all",
            "split",
        ]:
            # NOTE if "split", we assume that the context-target split has already been applied to the df
            # in an earlier scope with access to both the context and target data. This is maybe risky!
            X_c = df.reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_c = df.values.T
        elif isinstance(sampling_strat, np.ndarray):
            if df.index.get_level_values("x1").dtype != sampling_strat.dtype:
                raise InvalidSamplingStrategyError(
                    "Passed a numpy coordinate array to sample pandas DataFrame, "
                    "but the coordinate array has a different dtype than the DataFrame. "
                    f"Got {sampling_strat.dtype} but expected {df.index.get_level_values('x1').dtype}."
                )
            X_c = sampling_strat.astype(self.dtype)
            try:
                Y_c = df.loc[pd.IndexSlice[:, X_c[0], X_c[1]]].values.T
            except KeyError:
                raise InvalidSamplingStrategyError(
                    "Passed a numpy coordinate array to sample pandas DataFrame, "
                    "but the DataFrame did not contain all the requested samples.\n"
                    f"Indexes: {df.index}\n"
                    f"Sampling coords: {X_c}\n"
                    "If this is unexpected, check that your numpy sampling array matches "
                    "the DataFrame index values *exactly*."
                )
        else:
            raise InvalidSamplingStrategyError(
                f"Unknown sampling strategy {sampling_strat}"
            )

        if Y_c.ndim == 1:
            # returned a 1D array, but we need a 2D array of shape (variable, N)
            Y_c = Y_c.reshape(1, *Y_c.shape)

        return X_c, Y_c

    def sample_offgrid_aux(
        self,
        X_t: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        offgrid_aux: Union[xr.DataArray, xr.Dataset],
    ) -> np.ndarray:
        """Sample auxiliary data at off-grid locations.

        Args:
            X_t (:class:`numpy:numpy.ndarray` | Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]):
                Off-grid locations at which to sample the auxiliary data. Can
                be a tuple of two numpy arrays, or a single numpy array.
            offgrid_aux (:class:`xarray.DataArray` | :class:`xarray.Dataset`):
                Auxiliary data at off-grid locations.

        Returns:
            :class:`numpy:numpy.ndarray`:
                [Description of the returned numpy ndarray]

        Raises:
            [ExceptionType]:
                [Description of under what conditions this function raises an exception]
        """
        if "time" in offgrid_aux.dims:
            raise ValueError(
                "If `aux_at_targets` data has a `time` dimension, it must be sliced before "
                "passing it to `sample_offgrid_aux`."
            )
        if isinstance(X_t, tuple):
            xt1, xt2 = X_t
            xt1 = xt1.ravel()
            xt2 = xt2.ravel()
        else:
            xt1, xt2 = xr.DataArray(X_t[0]), xr.DataArray(X_t[1])
        Y_t_aux = offgrid_aux.sel(x1=xt1, x2=xt2, method="nearest")
        if isinstance(Y_t_aux, xr.Dataset):
            Y_t_aux = Y_t_aux.to_array()
        Y_t_aux = np.array(Y_t_aux, dtype=np.float32)
        if (isinstance(X_t, tuple) and Y_t_aux.ndim == 2) or (
            isinstance(X_t, np.ndarray) and Y_t_aux.ndim == 1
        ):
            # Reshape to (variable, *spatial_dims)
            Y_t_aux = Y_t_aux.reshape(1, *Y_t_aux.shape)
        return Y_t_aux

    def time_slice_variable(self, var, date, delta_t=0):
        """Slice a variable by a given time delta.

        Args:
            var (...):
                Variable to slice.
            delta_t (...):
                Time delta to slice by.

        Returns:
            var (...)
                Sliced variable.

        Raises:
            ValueError
                If the variable is of an unknown type.
        """
        # TODO: Does this work with instantaneous time?
        delta_t = pd.Timedelta(delta_t, unit=self.time_freq)
        if isinstance(var, (xr.Dataset, xr.DataArray)):
            if "time" in var.dims:
                var = var.sel(time=date + delta_t)
        elif isinstance(var, (pd.DataFrame, pd.Series)):
            if "time" in var.index.names:
                var = var[var.index.get_level_values("time") == date + delta_t]
        else:
            raise ValueError(f"Unknown variable type {type(var)}")
        return var

    def task_generation(     # noqa: D102
        self,
        date: pd.Timestamp,
        context_sampling: Union[
            str,
            int,
            float,
            np.ndarray,
            List[Union[str, int, float, np.ndarray]],
        ] = "all",
        target_sampling: Optional[
            Union[
                str,
                int,
                float,
                np.ndarray,
                List[Union[str, int, float, np.ndarray]],
            ]
        ] = None,
        split_frac: float = 0.5,
        datewise_deterministic: bool = False,
        seed_override: Optional[int] = None,
    ) -> Task:
        def check_sampling_strat(sampling_strat, set):
            """Check the sampling strategy.

            Ensure ``sampling_strat`` is either a single strategy (broadcast
            to all sets) or a list of length equal to the number of sets.
            Convert to a tuple of length equal to the number of sets and
            return.

            Args:
                sampling_strat:
                    Sampling strategy to check.
                set:
                    Context or target set to check.

            Returns:
                tuple:
                    Tuple of sampling strategies, one for each set.

            Raises:
                InvalidSamplingStrategyError:
                    - If the sampling strategy is invalid.
                    - If the length of the sampling strategy does not match the number of sets.
                    - If the sampling strategy is not a valid type.
                    - If the sampling strategy is a float but not in [0, 1].
                    - If the sampling strategy is an int but not positive.
                    - If the sampling strategy is a numpy array but not of shape (2, N).
            """
            if sampling_strat is None:
                return None
            if not isinstance(sampling_strat, (list, tuple)):
                sampling_strat = tuple([sampling_strat] * len(set))
            elif isinstance(sampling_strat, (list, tuple)) and len(
                sampling_strat
            ) != len(set):
                raise InvalidSamplingStrategyError(
                    f"Length of sampling_strat ({len(sampling_strat)}) must "
                    f"match number of context sets ({len(set)})"
                )

            for strat in sampling_strat:
                if not isinstance(strat, (str, int, np.integer, float, np.ndarray)):
                    raise InvalidSamplingStrategyError(
                        f"Unknown sampling strategy {strat} of type {type(strat)}"
                    )
                elif isinstance(strat, str) and strat not in [
                    "all",
                    "split",
                    "gapfill",
                ]:
                    raise InvalidSamplingStrategyError(
                        f"Unknown sampling strategy {strat} for type str"
                    )
                elif isinstance(strat, float) and not 0 <= strat <= 1:
                    raise InvalidSamplingStrategyError(
                        f"If sampling strategy is a float, must be fraction "
                        f"must be in [0, 1], got {strat}"
                    )
                elif isinstance(strat, int) and strat < 0:
                    raise InvalidSamplingStrategyError(
                        f"Sampling N must be positive, got {strat}"
                    )
                elif isinstance(strat, np.ndarray) and strat.shape[0] != 2:
                    raise InvalidSamplingStrategyError(
                        "Sampling coordinates must be of shape (2, N), got "
                        f"{strat.shape}"
                    )

            return sampling_strat

        def sample_variable(var, sampling_strat, seed):
            """Sample a variable by a given sampling strategy to get input and
            output data.

            Args:
                var:
                    Variable to sample.
                sampling_strat:
                    Sampling strategy to use.
                seed:
                    Seed for random sampling.

            Returns:
                Tuple[X, Y]:
                    Tuple of input and output data.

            Raises:
                ValueError:
                    If the variable is of an unknown type.
            """
            if isinstance(var, (xr.Dataset, xr.DataArray)):
                X, Y = self.sample_da(var, sampling_strat, seed)
            elif isinstance(var, (pd.DataFrame, pd.Series)):
                X, Y = self.sample_df(var, sampling_strat, seed)
            else:
                raise ValueError(f"Unknown type {type(var)} for context set " f"{var}")
            return X, Y

        # Check that the sampling strategies are valid
        context_sampling = check_sampling_strat(context_sampling, self.context)
        target_sampling = check_sampling_strat(target_sampling, self.target)
        # Check `split_frac
        if split_frac < 0 or split_frac > 1:
            raise ValueError(f"split_frac must be between 0 and 1, got {split_frac}")
        if self.links is None:
            b1 = any(
                [
                    strat in ["split", "gapfill"]
                    for strat in context_sampling
                    if isinstance(strat, str)
                ]
            )
            if target_sampling is None:
                b2 = False
            else:
                b2 = any(
                    [
                        strat in ["split", "gapfill"]
                        for strat in target_sampling
                        if isinstance(strat, str)
                    ]
                )
            if b1 or b2:
                raise ValueError(
                    "If using 'split' or 'gapfill' sampling strategies, the context and target "
                    "sets must be linked with the TaskLoader `links` attribute."
                )
        if self.links is not None:
            for context_idx, target_idx in self.links:
                context_sampling_i = context_sampling[context_idx]
                if target_sampling is None:
                    target_sampling_i = None
                else:
                    target_sampling_i = target_sampling[target_idx]
                link_strats = (context_sampling_i, target_sampling_i)
                if any(
                    [
                        strat in ["split", "gapfill"]
                        for strat in link_strats
                        if isinstance(strat, str)
                    ]
                ):
                    # If one of the sampling strategies is "split" or "gapfill", the other must
                    # use the same splitting strategy
                    if link_strats[0] != link_strats[1]:
                        raise ValueError(
                            f"Linked context set {context_idx} and target set {target_idx} "
                            f"must use the same sampling strategy if one of them "
                            f"uses the 'split' or 'gapfill' sampling strategy. "
                            f"Got {link_strats[0]} and {link_strats[1]}."
                        )

        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)

        if seed_override is not None:
            # Override the seed for random sampling
            seed = seed_override
        elif datewise_deterministic:
            # Generate a deterministic seed, based on the date, for random sampling
            seed = int(date.strftime("%Y%m%d"))
        else:
            # 'Truly' random sampling
            seed = None

        task = {}

        task["time"] = date
        task["ops"] = []
        task["X_c"] = []
        task["Y_c"] = []
        if target_sampling is not None:
            task["X_t"] = []
            task["Y_t"] = []
        else:
            task["X_t"] = None
            task["Y_t"] = None

        context_slices = [
            self.time_slice_variable(var, date, delta_t)
            for var, delta_t in zip(self.context, self.context_delta_t)
        ]
        target_slices = [
            self.time_slice_variable(var, date, delta_t)
            for var, delta_t in zip(self.target, self.target_delta_t)
        ]

        # TODO move to method
        if (
            self.links is not None
            and "split" in context_sampling
            and "split" in target_sampling
        ):
            # Perform the split sampling strategy for linked context and target sets at this point
            # while we have the full context and target data in scope

            context_split_idxs = np.where(np.array(context_sampling) == "split")[0]
            target_split_idxs = np.where(np.array(target_sampling) == "split")[0]
            assert len(context_split_idxs) == len(target_split_idxs), (
                f"Number of context sets with 'split' sampling strategy "
                f"({len(context_split_idxs)}) must match number of target sets "
                f"with 'split' sampling strategy ({len(target_split_idxs)})"
            )
            for split_i, (context_idx, target_idx) in enumerate(
                zip(context_split_idxs, target_split_idxs)
            ):
                assert (context_idx, target_idx) in self.links, (
                    f"Context set {context_idx} and target set {target_idx} must be linked, "
                    f"with the `links` attribute if using the 'split' sampling strategy"
                )

                context_var = context_slices[context_idx]
                target_var = target_slices[target_idx]

                for var in [context_var, target_var]:
                    assert isinstance(var, (pd.Series, pd.DataFrame)), (
                        f"If using 'split' sampling strategy for linked context and target sets, "
                        f"the context and target sets must be pandas DataFrames or Series, "
                        f"but got {type(var)}."
                    )

                N_obs = len(context_var)
                N_obs_target_check = len(target_var)
                if N_obs != N_obs_target_check:
                    raise ValueError(
                        f"Cannot split context set {context_idx} and target set {target_idx} "
                        f"because they have different numbers of observations: "
                        f"{N_obs} and {N_obs_target_check}"
                    )
                split_seed = seed + split_i if seed is not None else None
                rng = np.random.default_rng(split_seed)

                N_context = int(N_obs * split_frac)
                idxs_context = rng.choice(N_obs, N_context, replace=False)

                context_var = context_var.iloc[idxs_context]
                target_var = target_var.drop(context_var.index)

                context_slices[context_idx] = context_var
                target_slices[target_idx] = target_var

        # TODO move to method
        if (
            self.links is not None
            and "gapfill" in context_sampling
            and "gapfill" in target_sampling
        ):
            # Perform the gapfill sampling strategy for linked context and target sets at this point
            # while we have the full context and target data in scope

            context_gapfill_idxs = np.where(np.array(context_sampling) == "gapfill")[0]
            target_gapfill_idxs = np.where(np.array(target_sampling) == "gapfill")[0]
            assert len(context_gapfill_idxs) == len(target_gapfill_idxs), (
                f"Number of context sets with 'gapfill' sampling strategy "
                f"({len(context_gapfill_idxs)}) must match number of target sets "
                f"with 'gapfill' sampling strategy ({len(target_gapfill_idxs)})"
            )
            for gapfill_i, (context_idx, target_idx) in enumerate(
                zip(context_gapfill_idxs, target_gapfill_idxs)
            ):
                assert (context_idx, target_idx) in self.links, (
                    f"Context set {context_idx} and target set {target_idx} must be linked, "
                    f"with the `links` attribute if using the 'gapfill' sampling strategy"
                )

                context_var = context_slices[context_idx]
                target_var = target_slices[target_idx]

                for var in [context_var, target_var]:
                    assert isinstance(var, (xr.DataArray, xr.Dataset)), (
                        f"If using 'gapfill' sampling strategy for linked context and target sets, "
                        f"the context and target sets must be xarray DataArrays or Datasets, "
                        f"but got {type(var)}."
                    )

                split_seed = seed + gapfill_i if seed is not None else None
                rng = np.random.default_rng(split_seed)

                # Keep trying until we get a target set with at least one target point
                keep_searching = True
                while keep_searching:
                    added_mask_date = rng.choice(self.context[context_idx].time)
                    added_mask = (
                        self.context[context_idx].sel(time=added_mask_date).isnull()
                    )
                    curr_mask = context_var.isnull()

                    # Mask out added missing values
                    context_var = context_var.where(~added_mask)

                    # TEMP: Inefficient to convert all non-targets to NaNs and then remove NaNs
                    #   when we could just slice the target values here
                    target_mask = added_mask & ~curr_mask
                    if isinstance(target_var, xr.Dataset):
                        keep_searching = np.all(target_mask.to_array().data == False)
                    else:
                        keep_searching = np.all(target_mask.data == False)
                    if keep_searching:
                        continue  # No target points -- use a different `added_mask`

                    target_var = target_var.where(
                        target_mask
                    )  # Only keep target locations

                    context_slices[context_idx] = context_var
                    target_slices[target_idx] = target_var

        for i, (var, sampling_strat) in enumerate(
            zip(context_slices, context_sampling)
        ):
            context_seed = seed + i if seed is not None else None
            X_c, Y_c = sample_variable(var, sampling_strat, context_seed)
            task[f"X_c"].append(X_c)
            task[f"Y_c"].append(Y_c)
        if target_sampling is not None:
            for j, (var, sampling_strat) in enumerate(
                zip(target_slices, target_sampling)
            ):
                target_seed = seed + i + j if seed is not None else None
                X_t, Y_t = sample_variable(var, sampling_strat, target_seed)
                task[f"X_t"].append(X_t)
                task[f"Y_t"].append(Y_t)

        if self.aux_at_contexts is not None:
            # Add auxiliary variable sampled at context set as a new context variable
            X_c_offgrid = [X_c for X_c in task["X_c"] if not isinstance(X_c, tuple)]
            if len(X_c_offgrid) == 0:
                # No offgrid context sets
                X_c_offrid_all = np.empty((2, 0), dtype=self.dtype)
            else:
                X_c_offrid_all = np.concatenate(X_c_offgrid, axis=1)
            Y_c_aux = (
                self.sample_offgrid_aux(
                    X_c_offrid_all,
                    self.time_slice_variable(self.aux_at_contexts, date),
                ),
            )
            task["X_c"].append(X_c_offrid_all)
            task["Y_c"].append(Y_c_aux)

        if self.aux_at_targets is not None and target_sampling is None:
            task["Y_t_aux"] = None
        elif self.aux_at_targets is not None and target_sampling is not None:
            # Add auxiliary variable to target set
            if len(task["X_t"]) > 1:
                raise ValueError(
                    "Cannot add auxiliary variable to target set when there "
                    "are multiple target variables (not supported by default `ConvNP` model)."
                )
            task["Y_t_aux"] = self.sample_offgrid_aux(
                task["X_t"][0],
                self.time_slice_variable(self.aux_at_targets, date),
            )

        return Task(task)

    def __call__(
        self,
        date: pd.Timestamp,
        context_sampling: Union[
            str,
            int,
            float,
            np.ndarray,
            List[Union[str, int, float, np.ndarray]],
        ] = "all",
        target_sampling: Optional[
            Union[
                str,
                int,
                float,
                np.ndarray,
                List[Union[str, int, float, np.ndarray]],
            ]
        ] = None,
        split_frac: float = 0.5,
        datewise_deterministic: bool = False,
        seed_override: Optional[int] = None,
    ) -> Union[Task, List[Task]]:
        """Generate a task for a given date (or a list of
        :class:`.data.task.Task` objects for a list of dates).

        There are several sampling strategies available for the context and
        target data:

            - "all": Sample all observations.
            - int: Sample N observations uniformly at random.
            - float: Sample a fraction of observations uniformly at random.
            - :class:`numpy:numpy.ndarray`, shape (2, N):
                Sample N observations at the given x1, x2 coordinates. Coords are assumed to be
                normalised.
            - "split": Split pandas observations into disjoint context and target sets.
                `split_frac` determines the fraction of observations
                to use for the context set. The remaining observations are used
                for the target set.
                The context set and target set must be linked through the ``TaskLoader``
                ``links`` argument. Only valid for pandas data.
            - "gapfill": Generates a training task for filling NaNs in xarray data.
                Randomly samples a missing data (NaN) mask from another timestamp and
                adds it to the context set (i.e. increases the number of NaNs).
                The target set is then true values of the data at the added missing locations.
                The context set and target set must be linked through the ``TaskLoader``
                ``links`` argument. Only valid for xarray data.

        Args:
            date (:class:`pandas.Timestamp`):
                Date for which to generate the task.
            context_sampling (str | int | float | :class:`numpy:numpy.ndarray` | List[str | int | float | :class:`numpy:numpy.ndarray`], optional):
                Sampling strategy for the context data, either a list of
                sampling strategies for each context set, or a single strategy
                applied to all context sets. Default is ``"all"``.
            target_sampling (str | int | float | :class:`numpy:numpy.ndarray` | List[str | int | float | :class:`numpy:numpy.ndarray`], optional):
                Sampling strategy for the target data, either a list of
                sampling strategies for each target set, or a single strategy
                applied to all target sets. Default is ``None``, meaning no target
                data is returned.
            split_frac (float, optional):
                The fraction of observations to use for the context set with
                the "split" sampling strategy for linked context and target set
                pairs. The remaining observations are used for the target set.
                Default is 0.5.
            datewise_deterministic (bool, optional):
                Whether random sampling is datewise deterministic based on the
                date. Default is ``False``.
            seed_override (Optional[int], optional):
                Override the seed for random sampling. This can be used to use
                the same random sampling at different ``date``. Default is
                None.

        Returns:
            :class:`~.data.task.Task` | List[:class:`~.data.task.Task`]:
                Task object or list of task objects for each date containing
                the context and target data.
        """
        if isinstance(date, (list, tuple, pd.core.indexes.datetimes.DatetimeIndex)):
            return [
                self.task_generation(
                    d,
                    context_sampling,
                    target_sampling,
                    split_frac,
                    datewise_deterministic,
                    seed_override,
                )
                for d in date
            ]
        else:
            return self.task_generation(
                date,
                context_sampling,
                target_sampling,
                split_frac,
                datewise_deterministic,
                seed_override,
            )
