from deepsensor.data.task import Task

import numpy as np
import xarray as xr
import pandas as pd

from typing import List, Union


def construct_x1x2_ds(gridded_ds):
    """
    Construct an xr.Dataset containing two vars, where each var is a 2D gridded channel whose
    values contain the x_1 and x_2 coordinate values, respectively.
    """
    X1, X2 = np.meshgrid(gridded_ds.x1, gridded_ds.x2, indexing="ij")
    ds = xr.Dataset(
        coords={"x1": gridded_ds.x1, "x2": gridded_ds.x2},
        data_vars={
            "x1_arr": (("x1", "x2"), X1),
            "x2_arr": (("x1", "x2"), X2),
        },
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
        context_delta_t: Union[int, List[int]] = 0,
        target_delta_t: Union[int, List[int]] = 0,
        time_freq: str = "D",
        dtype="float32",
    ) -> None:
        """Initialise a TaskLoader object

        :param context: Context data. Can be a single xr.DataArray, xr.Dataset or pd.DataFrame,
            or a list/tuple of these.
        :param target: Target data. Can be a single xr.DataArray, xr.Dataset or pd.DataFrame,
            or a list/tuple of these.
        :param context_delta_t: Time difference between context data and t=0 (task init time).
            Can be a single int (same for all context data) or a list/tuple of ints.
        :param target_delta_t: Time difference between target data and t=0 (task init time).
            Can be a single int (same for all target data) or a list/tuple of ints.
        :param time_freq: Time frequency of the data. Default: 'D' (daily).
        :param dtype: Data type of the data. Used to cast the data to the specified dtype.
            Default: 'float32'.
        """
        self.time_freq = time_freq
        self.dtype = dtype

        if isinstance(context, (xr.DataArray, xr.Dataset, pd.DataFrame)):
            context = (context,)
        if isinstance(target, (xr.DataArray, xr.Dataset, pd.DataFrame)):
            target = (target,)
        context, target = self.cast_context_and_target_to_dtype(context, target)
        self.context = context
        self.target = target

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

        self.context_dims, self.target_dims = self.count_context_and_target_data_dims()
        (
            self.context_var_IDs,
            self.target_var_IDs,
            self.context_var_IDs_and_delta_t,
            self.target_var_IDs_and_delta_t,
        ) = self.infer_context_and_target_var_IDs()

    def cast_context_and_target_to_dtype(
        self,
        context: List,
        target: List,
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

        return context, target

    def load_dask(self) -> None:
        """Load any `dask` data into memory"""

        def load(datasets):
            for i, var in enumerate(datasets):
                if isinstance(var, (xr.DataArray, xr.Dataset)):
                    var = var.load()

        load(self.context)
        load(self.target)

        return None

    def count_context_and_target_data_dims(self):
        """Count the number of data dimensions in the context and target data.

        Returns
        -------
        context_dims : tuple. Tuple of data dimensions in the context data.
        target_dims : tuple. Tuple of data dimensions in the target data.
        """

        def count_data_dims_of_tuple_of_sets(datasets):
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
            return tuple(dims)

        context_dims = count_data_dims_of_tuple_of_sets(self.context)
        target_dims = count_data_dims_of_tuple_of_sets(self.target)

        return context_dims, target_dims

    def infer_context_and_target_var_IDs(self):
        """Infer the variable IDs of the context and target data.

        Returns
        -------
        context_var_IDs : tuple. Tuple of variable IDs in the context data.
        target_var_IDs : tuple. Tuple of variable IDs in the target data.
        """

        def infer_var_IDs_of_tuple_of_sets(datasets, delta_ts=None):
            """If delta_ts is not None, then add the delta_t to the variable ID"""
            var_IDs = []
            # Distinguish between xr.DataArray, xr.Dataset and pd.DataFrame
            for i, var, in enumerate(datasets):
                if isinstance(var, xr.DataArray):
                    var_ID = (var.name,)  # Single data variable
                elif isinstance(var, xr.Dataset):
                    var_ID = tuple(var.data_vars.keys())  # Multiple data variables
                elif isinstance(var, pd.DataFrame):
                    var_ID = tuple(var.columns)
                else:
                    raise ValueError(f"Unknown type {type(var)} for context set {var}")

                if delta_ts is not None:
                    # Add delta_t to the variable ID
                    var_ID = tuple([f"{var_ID_i}_t{delta_ts[i]}" for var_ID_i in var_ID])
                else:
                    var_ID = tuple([f"{var_ID_i}" for var_ID_i in var_ID])

                var_IDs.append(var_ID)

            return tuple(var_IDs)

        context_var_IDs = infer_var_IDs_of_tuple_of_sets(self.context)
        context_var_IDs_and_delta_t = infer_var_IDs_of_tuple_of_sets(
            self.context, self.context_delta_t
        )
        target_var_IDs = infer_var_IDs_of_tuple_of_sets(self.target)
        target_var_IDs_and_delta_t = infer_var_IDs_of_tuple_of_sets(
            self.target, self.target_delta_t
        )

        return (
            context_var_IDs,
            target_var_IDs,
            context_var_IDs_and_delta_t,
            target_var_IDs_and_delta_t,
        )

    def __repr__(self):
        """Representation of the TaskLoader object (for developers)

        TODO make this a more verbose version of __str__
        """
        s = f"TaskLoader({len(self.context)} context sets, {len(self.target)} target sets)"
        s += f"\nContext variable IDs: {self.context_var_IDs_and_delta_t}"
        s += f"\nTarget variable IDs: {self.target_var_IDs_and_delta_t}"
        s += f"\nContext data dimensions: {self.context_dims}"
        s += f"\nTarget data dimensions: {self.target_dims}"
        return s

    def __str__(self):
        """String representation of the TaskLoader object (user-friendly)"""
        s = f"TaskLoader({len(self.context)} context sets, {len(self.target)} target sets)"
        s += f"\nContext variable IDs: {self.context_var_IDs}"
        s += f"\nTarget variable IDs: {self.target_var_IDs}"
        return s

    def sample_da(
        self,
        da: Union[xr.DataArray, xr.Dataset],
        sampling_strat: Union[str, int],
        seed: int = None,
    ) -> (np.ndarray, np.ndarray):
        """Sample a DataArray according to a given strategy

        :param da: DataArray to sample, assumed to be sliced for the task already
        :param sampling_strat: Sampling strategy, either "grid" or an integer for random grid cell sampling
        :param seed: Seed for random sampling
        :return: Sampled DataArray
        """
        da = da.load()  # Converts dask -> numpy if not already loaded
        if isinstance(da, xr.Dataset):
            da = da.to_array()

        if isinstance(sampling_strat, int):
            N = sampling_strat
            rng = np.random.default_rng(seed)
            x1 = rng.choice(da.coords["x1"].values, N)
            x2 = rng.choice(da.coords["x2"].values, N)
            X_c = np.array([x1, x2])
            Y_c = da.sel(x1=xr.DataArray(x1), x2=xr.DataArray(x2)).data
        elif sampling_strat == "grid":
            X_c = (da.coords["x1"].values, da.coords["x2"].values)
            Y_c = da.data
            if Y_c.ndim == 2:
                # "grid" sampling returned a 2D array, but we need a 3D array of shape (variable, x1, x2)
                Y_c = Y_c.reshape(1, *Y_c.shape)
        else:
            raise ValueError(f"Unknown sampling strategy {sampling_strat}")

        return X_c, Y_c

    def sample_df(
        self,
        df: Union[pd.DataFrame, pd.Series],
        sampling_strat: Union[str, int],
        seed: int = None,
    ) -> (np.ndarray, np.ndarray):
        """Sample a DataArray according to a given strategy

        :param da: DataArray to sample, assumed to be sliced for the task already
        :param sampling_strat: Sampling strategy, either "grid" or an integer for random grid cell sampling
        :param seed: Seed for random sampling
        :return: Sampled DataArray
        """
        df = df.dropna(how="any")  # If any obs are NaN, drop them
        if isinstance(sampling_strat, int):
            # N = sampling_strat
            # rng = np.random.default_rng()
            # x1 = rng.choice(da.coords["x1"].values, N)
            # x2 = rng.choice(da.coords["x2"].values, N)
            # X_c = np.array([x1, x2])
            # Y_c = da.sel(x1=xr.DataArray(x1), x2=xr.DataArray(x2)).data
            pass
        elif sampling_strat == "grid":  # TODO rename from "grid" to "all"
            # X_c = np.array([
            #     df.index.get_level_values("x1").values,
            #     df.index.get_level_values("x2").values,
            # ])
            X_c = df.reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_c = df.values.T
        else:
            raise ValueError(f"Unknown sampling strategy {sampling_strat}")

        return X_c, Y_c

    def task_generation(
        self,
        date: pd.Timestamp,
        context_sampling: Union[str, int, List[Union[str, int]]] = "grid",
        target_sampling: Union[str, int, List[Union[str, int]]] = "grid",
        deterministic: bool = False,
    ) -> Task:
        """Generate a task for a given date

        There are several sampling strategies available for the context and target data:
        - "grid": Sample all grid cells.
        - int: Sample N grid cells uniformly at random.

        :param date: Date for which to generate the task
        :param context_sampling: Sampling strategy for the context data, either a list of
            sampling strategies for each context set, or a single strategy applied to all context sets
        :param target_sampling: Sampling strategy for the target data, either a list of
            sampling strategies for each target set, or a single strategy applied to all target sets
        :param: deterministic: Whether random sampling is deterministic based on
        :return: Task object containing the context and target data
        """

        def check_sampling_strat(sampling_strat, set):
            """Check the sampling strategy fits the number of context sets and convert to tuple"""
            if not isinstance(sampling_strat, (list, tuple)):
                sampling_strat = tuple([sampling_strat] * len(set))
            elif isinstance(sampling_strat, (list, tuple)) and len(
                sampling_strat
            ) != len(set):
                raise ValueError(
                    f"Length of sampling_strat ({len(sampling_strat)}) must match number of"
                    f"context sets ({len(set)})"
                )
            elif isinstance(sampling_strat, (str, int)):
                sampling_strat = tuple([sampling_strat] * len(set))
            return sampling_strat

        context_sampling = check_sampling_strat(context_sampling, self.context)
        target_sampling = check_sampling_strat(target_sampling, self.target)

        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)

        if deterministic:
            # Generate a deterministic seed, based on the date, for random sampling
            seed = int(date.strftime("%Y%m%d"))
        else:
            # 'Truly' random sampling
            seed = None

        task = {}

        task["time"] = date
        task[
            "flag"
        ] = None  # Flag for modifying the task (reshaping, adding data, etc.)
        task["X_c"] = []
        task["Y_c"] = []
        task["X_t"] = []
        task["Y_t"] = []

        def sample_variable(var, sampling_strat, delta_t, seed):
            delta_t = pd.Timedelta(delta_t, unit=self.time_freq)
            if isinstance(var, (xr.Dataset, xr.DataArray)):
                if "time" in var.dims:
                    var = var.sel(time=date + delta_t)
                X_c, Y_c = self.sample_da(var, sampling_strat, seed)
            elif type(var) is pd.DataFrame:
                if "time" in var.index.names:
                    var = var.loc[date + delta_t]
                X_c, Y_c = self.sample_df(var, sampling_strat, seed)
            else:
                raise ValueError(f"Unknown type {type(var)} for context set {var}")

            return X_c, Y_c

        for i, (var, sampling_strat, delta_t) in enumerate(
            zip(self.context, context_sampling, self.context_delta_t)
        ):
            context_seed = seed + i if seed is not None else None
            X_c, Y_c = sample_variable(var, sampling_strat, delta_t, context_seed)
            task[f"X_c"].append(X_c)
            task[f"Y_c"].append(Y_c)
        for i, (var, sampling_strat, delta_t) in enumerate(
            zip(self.target, target_sampling, self.target_delta_t)
        ):
            target_seed = seed + i if seed is not None else None
            X_t, Y_t = sample_variable(var, sampling_strat, delta_t, target_seed)
            task[f"X_t"].append(X_t)
            task[f"Y_t"].append(Y_t)

        return Task(task)

    def __call__(self, date, *args, **kwargs):
        if isinstance(date, (list, tuple, pd.core.indexes.datetimes.DatetimeIndex)):
            return [self.task_generation(d, *args, **kwargs) for d in date]
        else:
            return self.task_generation(date, *args, **kwargs)
