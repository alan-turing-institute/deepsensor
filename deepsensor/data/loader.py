from deepsensor.data.task import Task, flatten_X

import numpy as np
import xarray as xr
import pandas as pd

from typing import List, Tuple, Union

from deepsensor.errors import InvalidSamplingStrategyError


class TaskLoader:
    def __init__(
        self,
        context: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
        ],
        target: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
        ],
        aux_at_contexts: Union[xr.DataArray, xr.Dataset] = None,
        aux_at_targets: Union[
            xr.DataArray,
            xr.Dataset,
        ] = None,
        links: Union[Tuple, List[Tuple[int, int]], None] = None,
        context_delta_t: Union[int, List[int]] = 0,
        target_delta_t: Union[int, List[int]] = 0,
        time_freq: str = "D",
        xarray_interp_method: str = "linear",
        discrete_xarray_sampling: bool = False,
        dtype: object = "float32",
    ) -> None:
        """Initialise a TaskLoader object

        Args:
            context: Context data. Can be a single xr.DataArray, xr.Dataset or pd.DataFrame,
                or a list/tuple of these.
            target: Target data. Can be a single xr.DataArray, xr.Dataset or pd.DataFrame,
                or a list/tuple of these.
            aux_at_contexts: Gridded auxiliary data to sample at off-grid context locations. Can be a
                single xr.DataArray or xr.Dataset object. This xarray object is automatically sampled at the
                locations of any off-grid context sets and these extra observations are passed via
                an additional context set. Default: None.
            aux_at_targets: Gridded auxiliary data to sample at target locations. Can be a single
                xr.DataArray or xr.Dataset. This xarray object is automatically sampled at the
                target locations and these extra observations are included as a separate `Y_t_aux`
                entry in the Task object. Not supported for multiple target sets. Default: None.
            links: Specifies links between context and target data. Each link is a tuple of
                two integers, where the first integer is the index of the context data and the second
                integer is the index of the target data. Can be a single tuple in the case of a single
                link. If None, no links are specified. Default: None.
            context_delta_t: Time difference between context data and t=0 (task init time).
                Can be a single int (same for all context data) or a list/tuple of ints.
            target_delta_t: Time difference between target data and t=0 (task init time).
                Can be a single int (same for all target data) or a list/tuple of ints.
            time_freq: Time frequency of the data. Default: 'D' (daily).
            xarray_interp_method: Interpolation method to use when interpolating xr.DataArray
            discrete_xarray_sampling: When randomly sampling xarray variables, whether to sample
                at discrete points defined at grid cell centres, or at continuous points within the grid.
                Default is False.
            dtype: Data type of the data. Used to cast the data to the specified dtype.
                Default: 'float32'.
        """
        self.time_freq = time_freq
        self.xarray_interp_method = xarray_interp_method
        self.discrete_xarray_sampling = discrete_xarray_sampling
        self.dtype = dtype

        if aux_at_contexts is not None:
            self._check_offgrid_aux(self._check_offgrid_aux(aux_at_contexts))
        if aux_at_targets is not None:
            self._check_offgrid_aux(aux_at_targets)

        if isinstance(context, (xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series)):
            context = (context,)
        if isinstance(target, (xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series)):
            target = (target,)
        (
            context,
            target,
            aux_at_contexts,
            aux_at_targets,
        ) = self._cast_context_and_target_to_dtype(
            context, target, aux_at_contexts, aux_at_targets
        )
        self.context = context
        self.target = target
        self.aux_at_contexts = aux_at_contexts
        self.aux_at_targets = aux_at_targets

        self.links = self.check_links(links)

        if isinstance(context_delta_t, int):
            context_delta_t = (context_delta_t,) * len(context)
        else:
            assert len(context_delta_t) == len(context)
        if isinstance(target_delta_t, int):
            target_delta_t = (target_delta_t,) * len(target)
        else:
            assert len(target_delta_t) == len(target)
        self.context_delta_t = context_delta_t
        self.target_delta_t = target_delta_t

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

    def _cast_context_and_target_to_dtype(
        self,
        context: List,
        target: List,
        aux_at_contexts: Union[xr.DataArray, xr.Dataset] = None,
        aux_at_targets: Union[xr.DataArray, xr.Dataset] = None,
    ) -> (List, List):
        """Cast context and target data to the default dtype.

        Returns
        -------
        context : tuple. Tuple of context data with specified dtype.
        target : tuple. Tuple of target data with specified dtype.
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
                # x1/x2 coord dtypes during task sampling
            else:
                raise ValueError(f"Unknown type {type(var)} for context set {var}")
            return var

        context = tuple([cast_to_dtype(var) for var in context])
        target = tuple([cast_to_dtype(var) for var in target])
        if aux_at_contexts is not None:
            aux_at_contexts = aux_at_contexts.astype(self.dtype)
        if aux_at_targets is not None:
            aux_at_targets = aux_at_targets.astype(self.dtype)

        return context, target, aux_at_contexts, aux_at_targets

    def _check_offgrid_aux(self, offgrid_aux):
        if offgrid_aux is not None and "time" in offgrid_aux.dims:
            raise ValueError(
                "Auxiliary data has a time dimension. Spatiotemporal auxiliary data is not yet supported. "
                "Please slice the auxiliary data to a single time step."
            )

    def load_dask(self) -> None:
        """Load any `dask` data into memory"""

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

        Returns
        -------
        context_dims : tuple. Tuple of data dimensions in the context data.
        target_dims : tuple. Tuple of data dimensions in the target data.
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

        Returns
        -------
        context_var_IDs : tuple. Tuple of variable IDs in the context data.
        target_var_IDs : tuple. Tuple of variable IDs in the target data.
        """

        def infer_var_IDs_of_tuple_of_sets(datasets, delta_ts=None):
            """If delta_ts is not None, then add the delta_t to the variable ID"""
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

    def check_links(self, links):
        """Check that the context-target links are valid."""
        if links is None:
            return None

        if type(links) is list and type(links[0]) is not tuple:
            raise ValueError("If `links` is a list, then it must be a list of tuples")
        elif type(links) is tuple and type(links[0]) is not int:
            raise ValueError("If `links` is a tuple, then it must be a tuple of ints")
        elif type(links) is tuple and len(links) != 2:
            raise ValueError("If `links` is a tuple of ints, then it must be length 2")
        elif type(links) is tuple and len(links) == 2:
            # Convert single tuple to list of tuples
            links = [links]

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
            if not isinstance(self.context[context_idx], (pd.DataFrame, pd.Series)):
                raise ValueError(
                    f"Context set {context_idx} must be a pandas object when using the 'split' sampling strategy"
                )
            if not isinstance(self.target[target_idx], (pd.DataFrame, pd.Series)):
                raise ValueError(
                    f"Target set {target_idx} must be a pandas object when using the 'split' sampling strategy"
                )

        return links

    def __str__(self):
        """String representation of the TaskLoader object (user-friendly)"""
        s = f"TaskLoader({len(self.context_dims)} context sets, {len(self.target_dims)} target sets)"
        s += f"\nContext variable IDs: {self.context_var_IDs}"
        s += f"\nTarget variable IDs: {self.target_var_IDs}"
        if self.aux_at_targets is not None:
            s += f"\nAuxiliary-at-target variable IDs: {self.aux_at_target_var_IDs}"
        return s

    def __repr__(self):
        """Representation of the TaskLoader object (for developers)

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
        seed: int = None,
    ) -> (np.ndarray, np.ndarray):
        """Sample a DataArray according to a given strategy

        :param da: DataArray to sample, assumed to be sliced for the task already
        :param sampling_strat: Sampling strategy, either "all" or an integer for random grid cell sampling
        :param seed: Seed for random sampling
        :return: Sampled DataArray
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
                    if isinstance(da, xr.Dataset):
                        dim = len(da.data_vars)  # Multiple data variables
                    elif isinstance(da, xr.DataArray):
                        dim = 1  # Single data variable
                    Y_c = np.zeros((dim, 0), dtype=self.dtype)
                    return X_c, Y_c
                x1 = rng.uniform(da.coords["x1"].min(), da.coords["x1"].max(), N)
                x2 = rng.uniform(da.coords["x2"].min(), da.coords["x2"].max(), N)
                Y_c = da.interp(
                    x1=xr.DataArray(x1),
                    x2=xr.DataArray(x2),
                    method=self.xarray_interp_method,
                    kwargs=dict(fill_value=None, bounds_error=True),
                )
                Y_c = np.array(Y_c, dtype=self.dtype)
            X_c = np.array([x1, x2], dtype=self.dtype)
            if Y_c.ndim == 1:
                # returned a 1D array, but we need a 2D array of shape (variable, N)
                Y_c = Y_c.reshape(1, *Y_c.shape)

        elif isinstance(sampling_strat, np.ndarray):
            X_c = sampling_strat.astype(self.dtype)
            try:
                Y_c = da.interp(
                    x1=xr.DataArray(X_c[0]),
                    x2=xr.DataArray(X_c[1]),
                    method=self.xarray_interp_method,
                    kwargs=dict(fill_value=None, bounds_error=True),
                )
            except ValueError:
                raise InvalidSamplingStrategyError(
                    f"Passed a numpy coordinate array to sample xarray object, "
                    f"but the coordinates are out of bounds."
                )
            Y_c = np.array(Y_c, dtype=self.dtype)
            if Y_c.ndim == 1:
                # returned a 1D array, but we need a 2D array of shape (variable, N)
                Y_c = Y_c.reshape(1, *Y_c.shape)

        elif sampling_strat == "all":
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

        return X_c, Y_c

    def sample_df(
        self,
        df: Union[pd.DataFrame, pd.Series],
        sampling_strat: Union[str, int, float, np.ndarray],
        seed: int = None,
    ) -> (np.ndarray, np.ndarray):
        """Sample a DataArray according to a given strategy

        :param da: DataArray to sample, assumed to be time-sliced for the task already
        :param sampling_strat: Sampling strategy, either "all" or an integer for random grid cell sampling
        :param seed: Seed for random sampling
        :return: Sampled DataArray
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
        elif sampling_strat in ["all", "split"]:
            # NOTE if "split", we assume that the context-target split has already been applied to the df
            # in an earlier scope with access to both the context and target data. This is maybe risky!
            X_c = df.reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_c = df.values.T
        elif isinstance(sampling_strat, np.ndarray):
            X_c = sampling_strat.astype(self.dtype)
            x1match = np.in1d(df.index.get_level_values("x1"), X_c[0])
            x2match = np.in1d(df.index.get_level_values("x2"), X_c[1])
            num_matches = np.sum(x1match & x2match)

            # Check that we got all the samples we asked for
            if num_matches != X_c.shape[1]:
                raise InvalidSamplingStrategyError(
                    f"Passed a numpy coordinate array to sample pandas DataFrame, "
                    f"but the DataFrame did not contain all the requested samples. "
                    f"Requested {X_c.shape[1]} samples but only got {num_matches}."
                )

            Y_c = df[x1match & x2match].values.T
        else:
            raise InvalidSamplingStrategyError(
                f"Unknown sampling strategy {sampling_strat}"
            )

        return X_c, Y_c

    def sample_offgrid_aux(
        self,
        X_t: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        offgrid_aux: Union[xr.DataArray, xr.Dataset],
    ) -> np.ndarray:
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

    def task_generation(
        self,
        date: pd.Timestamp,
        context_sampling: Union[
            str, int, float, np.ndarray, List[Union[str, int, float, np.ndarray]]
        ] = "all",
        target_sampling: Union[
            str, int, float, np.ndarray, List[Union[str, int, float, np.ndarray]]
        ] = "all",
        split_frac: float = 0.5,
        datewise_deterministic: bool = False,
        seed_override=None,
    ) -> Task:
        """Generate a task for a given date

        There are several sampling strategies available for the context and target data:
        - "all": Sample all observations.
        - int: Sample N observations uniformly at random.
        - float: Sample a fraction of observations uniformly at random.
        - np.ndarray, shape (2, N): Sample N observations at the given x1, x2 coordinates.
            Coords are assumed to be unnormalised.

        :param date: Date for which to generate the task
        :param context_sampling: Sampling strategy for the context data, either a list of
            sampling strategies for each context set, or a single strategy applied to all context sets
        :param target_sampling: Sampling strategy for the target data, either a list of
            sampling strategies for each target set, or a single strategy applied to all target sets
        :param split_frac: The fraction of observations to use for the context set with the "split"
            sampling strategy for linked context and target set pairs. The remaining observations
            are used for the target set. Default is 0.5.
        :param: datewise_deterministic: Whether random sampling is datewise_deterministic based on the date. Default is False.
        :param seed_override: Override the seed for random sampling. This can be used to use the same
            random sampling at different `date`s. Default is None.
        :return: Task object containing the context and target data
        """

        def check_sampling_strat(sampling_strat, set):
            """Check the sampling strategy

            Ensure `sampling_strat` is either a single strategy (broadcast to all sets) or a list
            of length equal to the number of sets. Convert to a tuple of length equal to the number
            of sets and return.
            """
            if not isinstance(sampling_strat, (list, tuple)):
                sampling_strat = tuple([sampling_strat] * len(set))
            elif isinstance(sampling_strat, (list, tuple)) and len(
                sampling_strat
            ) != len(set):
                raise InvalidSamplingStrategyError(
                    f"Length of sampling_strat ({len(sampling_strat)}) must match number of "
                    f"context sets ({len(set)})"
                )

            for strat in sampling_strat:
                if not isinstance(strat, (str, int, np.integer, float, np.ndarray)):
                    raise InvalidSamplingStrategyError(
                        f"Unknown sampling strategy {strat} of type {type(strat)}"
                    )
                elif isinstance(strat, str) and strat not in ["all", "split"]:
                    raise InvalidSamplingStrategyError(
                        f"Unknown sampling strategy {strat} for type str"
                    )
                elif isinstance(strat, float) and not 0 <= strat <= 1:
                    raise InvalidSamplingStrategyError(
                        f"If sampling strategy is a float, must be fraction must be in [0, 1], got {strat}"
                    )
                elif isinstance(strat, int) and strat < 0:
                    raise InvalidSamplingStrategyError(
                        f"Sampling N must be positive, got {strat}"
                    )
                elif isinstance(strat, np.ndarray) and strat.shape[0] != 2:
                    raise InvalidSamplingStrategyError(
                        f"Sampling coordinates must be of shape (2, N), got {strat.shape}"
                    )

            return sampling_strat

        def time_slice_variable(var, delta_t):
            """Slice a variable by a given time delta"""
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

        def sample_variable(var, sampling_strat, seed):
            """Sample a variable by a given sampling strategy to get input and output data"""
            if isinstance(var, (xr.Dataset, xr.DataArray)):
                X, Y = self.sample_da(var, sampling_strat, seed)
            elif isinstance(var, (pd.DataFrame, pd.Series)):
                X, Y = self.sample_df(var, sampling_strat, seed)
            else:
                raise ValueError(f"Unknown type {type(var)} for context set {var}")
            return X, Y

        # Check that the sampling strategies are valid
        context_sampling = check_sampling_strat(context_sampling, self.context)
        target_sampling = check_sampling_strat(target_sampling, self.target)
        # Check `split_frac
        if split_frac < 0 or split_frac > 1:
            raise ValueError(f"split_frac must be between 0 and 1, got {split_frac}")
        if (
            self.links is not None
            and "split" in context_sampling
            and "split" not in target_sampling
        ):
            raise ValueError(
                "Cannot use 'split' sampling strategy for context set and not target set"
            )
        elif (
            self.links is not None
            and "split" not in context_sampling
            and "split" in target_sampling
        ):
            raise ValueError(
                "Cannot use 'split' sampling strategy for target set and not context set"
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
        # Flag for modifying the task (reshaping, adding data, etc.)
        task["flag"] = None
        task["X_c"] = []
        task["Y_c"] = []
        task["X_t"] = []
        task["Y_t"] = []

        context_slices = [
            time_slice_variable(var, delta_t)
            for var, delta_t in zip(self.context, self.context_delta_t)
        ]
        target_slices = [
            time_slice_variable(var, delta_t)
            for var, delta_t in zip(self.target, self.target_delta_t)
        ]

        if (
            self.links is not None
            and "split" in context_sampling
            and "split" in target_sampling
        ):
            # Perform the split sampling strategy for linked context and target sets at this point
            # while we have the full context and target data in scope
            for link_i, link in enumerate(self.links):
                N_obs = len(context_slices[link[0]])
                N_obs_target_check = len(target_slices[link[1]])
                if N_obs != N_obs_target_check:
                    raise ValueError(
                        f"Context set {link[0]} has {N_obs} observations, but target set {link[1]}"
                        f"has {N_obs_target_check} observations"
                    )

                N_context = int(N_obs * split_frac)
                split_seed = seed + link_i if seed is not None else None
                rng = np.random.default_rng(split_seed)
                idxs_context = rng.choice(N_obs, N_context, replace=False)
                context_slices[link[0]] = context_slices[link[0]].iloc[idxs_context]
                target_slices[link[1]] = target_slices[link[1]].drop(
                    context_slices[link[0]].index
                )

        for i, (var, sampling_strat) in enumerate(
            zip(context_slices, context_sampling)
        ):
            context_seed = seed + i if seed is not None else None
            X_c, Y_c = sample_variable(var, sampling_strat, context_seed)
            task[f"X_c"].append(X_c)
            task[f"Y_c"].append(Y_c)
        for j, (var, sampling_strat) in enumerate(zip(target_slices, target_sampling)):
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
            Y_c_aux = self.sample_offgrid_aux(X_c_offrid_all, self.aux_at_contexts)
            task["X_c"].append(X_c_offrid_all)
            task["Y_c"].append(Y_c_aux)

        if self.aux_at_targets is not None:
            # Add auxiliary variable to target set
            if len(task["X_t"]) > 1:
                raise ValueError(
                    "Cannot add auxiliary variable to target set when there are multiple target variables"
                )
            task["Y_t_aux"] = self.sample_offgrid_aux(
                task["X_t"][0], self.aux_at_targets
            )

        return Task(task)

    def __call__(self, date, *args, **kwargs):
        if isinstance(date, (list, tuple, pd.core.indexes.datetimes.DatetimeIndex)):
            return [self.task_generation(d, *args, **kwargs) for d in date]
        else:
            return self.task_generation(date, *args, **kwargs)
