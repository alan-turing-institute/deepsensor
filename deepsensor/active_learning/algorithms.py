import copy

from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import (
    xarray_to_coord_array_normalised,
    mask_coord_array_normalised,
    da1_da2_same_grid,
    interp_da1_to_da2,
)
from deepsensor.model.model import DeepSensorModel, create_empty_spatiotemporal_xarray
from deepsensor.data.task import Task, append_obs_to_task
from deepsensor.active_learning.acquisition_fns import (
    AcquisitionFunction,
    AcquisitionFunctionParallel,
    AcquisitionFunctionOracle,
)

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

from typing import Union, List, Tuple


class GreedyAlgorithm:
    """Greedy algorithm for active learning"""

    def __init__(
        self,
        model: DeepSensorModel,
        X_s: Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, pd.Index],
        X_t: Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, pd.Index],
        X_s_mask: Union[xr.Dataset, xr.DataArray] = None,
        X_t_mask: Union[xr.Dataset, xr.DataArray] = None,
        N_new_context: int = 1,
        X_normalised: bool = False,
        model_infill_method: str = "mean",
        query_infill: xr.DataArray = None,
        proposed_infill: xr.DataArray = None,
        context_set_idx: int = 0,
        target_set_idx: int = 0,
        progress_bar: bool = False,
        min_or_max: str = "min",
        task_loader: TaskLoader = None,  # OPTIONAL for oracle acquisition functions only
        verbose: bool = False,
    ):
        """

        Arguments:
            model {DeepSensorModel} -- Trained model to use for proposing new context points.
            X_s {Union[xr.Dataset, xr.DataArray} -- Search coordinates.
            X_t {Union[xr.Dataset, xr.DataArray} -- Target coordinates.
            X_s_mask {Union[xr.Dataset, xr.DataArray], optional} -- Mask for search coordinates.
                If provided, only points where mask is True will be considered. Defaults to None.
        """
        if not isinstance(model, DeepSensorModel):
            raise ValueError(
                f"`model` must inherit from DeepSensorModel, but parent classes are {model.__class__.__bases__}"
            )

        self._validate_n_new_context(X_s, N_new_context)

        self.model = model
        self.N_new_context = N_new_context
        self.progress_bar = progress_bar
        self.model_infill_method = model_infill_method
        self.context_set_idx = context_set_idx
        self.target_set_idx = target_set_idx
        self.min_or_max = min_or_max
        self.task_loader = task_loader
        self.pbar = None

        self.X_s_mask = X_s_mask
        self.X_t_mask = X_t_mask

        if min_or_max not in ["min", "max"]:
            raise ValueError(
                f"min_or_max must be 'min' or 'max', not {self.min_or_max}"
            )

        self.x1_name = self.model.data_processor.norm_params["coords"]["x1"]["name"]
        self.x2_name = self.model.data_processor.norm_params["coords"]["x2"]["name"]

        # Normalised search and target coordinates
        if not X_normalised:
            X_t = model.data_processor.map_coords(X_t)
            X_s = model.data_processor.map_coords(X_s)
            if X_s_mask is not None:
                X_s_mask = model.data_processor.map_coords(X_s_mask)
            if X_t_mask is not None:
                X_t_mask = model.data_processor.map_coords(X_t_mask)

        self.X_s = X_s
        self.X_t = X_t

        # Interpolate masks onto search and target coords
        self.X_s_mask, self.X_t_mask = self._process_masks(X_s_mask, X_t_mask, X_s, X_t)

        # Interpolate overridden infill datasets at search points if necessary
        if query_infill is not None and not da1_da2_same_grid(query_infill, X_s):
            if verbose:
                print("query_infill not on search grid, interpolating.")
            query_infill = interp_da1_to_da2(query_infill, self.X_s)
        if proposed_infill is not None and not da1_da2_same_grid(proposed_infill, X_s):
            if verbose:
                print("proposed_infill not on search grid, interpolating.")
            proposed_infill = interp_da1_to_da2(proposed_infill, self.X_s)
        self.query_infill = query_infill
        self.proposed_infill = proposed_infill

        # Convert target coords to numpy arrays and assign to tasks
        if isinstance(X_t, (xr.Dataset, xr.DataArray)):
            # Targets on grid
            self.X_t_arr = xarray_to_coord_array_normalised(X_t)
            if self.X_t_mask is not None:
                # Remove points that lie outside the mask
                self.X_t_arr = mask_coord_array_normalised(self.X_t_arr, self.X_t_mask)
        elif isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index)):
            # Targets off-grid
            self.X_t_arr = X_t.reset_index()[["x1", "x2"]].values.T
        else:
            raise TypeError(f"Unsupported type for X_t: {type(X_t)}")

        # Construct search array
        if isinstance(X_s, (xr.Dataset, xr.DataArray)):
            X_s_arr = xarray_to_coord_array_normalised(X_s)
            if X_s_mask is not None:
                X_s_arr = mask_coord_array_normalised(X_s_arr, self.X_s_mask)
        self.X_s_arr = X_s_arr

        self.X_new = []  # List of new proposed context locations

    @classmethod
    def _validate_n_new_context(
        cls, X_s: Union[xr.Dataset, xr.DataArray], N_new_context: int
    ):
        if isinstance(X_s, (xr.Dataset, xr.DataArray)):
            if isinstance(X_s, xr.Dataset):
                X_s = X_s.to_array()
            N_s = X_s.shape[-2] * X_s.shape[-1]
        elif isinstance(X_s, (pd.DataFrame, pd.Series, pd.Index)):
            N_s = len(X_s)

        if not 0 < N_new_context < N_s:
            raise ValueError(
                f"Number of new context ({N_new_context}) must be greater than zero "
                f"and less than the number of search points ({N_s})"
            )

    @classmethod
    def _process_masks(cls, X_s_mask: xr.DataArray, X_t_mask: xr.DataArray, X_s, X_t):
        """Process masks by interpolating to X_s and X_t"""
        # TODO avoid repeated code
        if X_s_mask is not None:
            X_s_mask = X_s_mask.astype(float).interp_like(
                X_s, method="nearest", kwargs={"fill_value": 0}
            )
            X_s_mask.data = X_s_mask.data.astype(bool)
            X_s_mask.load()
        if X_t_mask is not None:
            X_t_mask = X_t_mask.astype(float).interp_like(
                X_t, method="nearest", kwargs={"fill_value": 0}
            )
            X_t_mask.data = X_t_mask.data.astype(bool)
            X_t_mask.load()

        return X_s_mask, X_t_mask

    def _get_times_from_tasks(self):
        """Get times from tasks"""
        times = [task["time"] for task in self.tasks]
        # Check for any repeats
        if len(times) != len(set(times)):
            # TODO unit test this
            raise ValueError(
                f"The {len(times)} tasks have duplicate times ({len(set(times))} "
                f"unique times)"
            )
        return times

    def _model_infill_at_search_points(
        self,
        X_s: Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, pd.Index],
    ):
        """
        Computes and sets the model infill y-values over whole search grid
        before running greedy optimisation. Results are returned with additional
        first axis, with size > 1 if model_infill_method == 'sample' or 'ar_sample_*', and
        acquisition function will be averaged over the samples in the first axis.
        If model_infill_method != 'sample', first axis size is 1 and the averaging
        is only over one value (i.e. no averaging).

        Infilled y-values at all placement search locations are appended to each dataset
        with the key (`'Y_model_infilled'`) for use during the placement search.

        Also adds a sample dimension to the context station observations, which will
        be looped over for MCMC sampling of the acquisition function importance
        values of the placement criterion.
        """
        if self.model_infill_method == "mean":
            infill_ds, _ = self.model.predict(
                self.tasks,
                X_s,
                X_t_is_normalised=True,
                unnormalise=False,
            )

        elif self.model_infill_method == "sample":
            # _, _, infill_ds = self.model.predict(
            #     self.tasks, X_s, X_t_normalised=True, unnormalise=False,
            #     n_samples=self.model_infill_samples,
            # )
            raise NotImplementedError("TODO")

        elif self.model_infill_method == "zeros":
            # TODO generate empty prediction xarray
            raise NotImplementedError("TODO")

        else:
            raise ValueError(
                f"Unsupported model_infill_method: {self.model_infill_method}"
            )

        return infill_ds

    def _sample_y_infill(self, infill_ds, time, x1, x2):
        """Sample infill values at a single location"""
        if isinstance(infill_ds, (xr.Dataset, xr.DataArray)):
            y = infill_ds.sel(time=time, x1=x1, x2=x2)
            if isinstance(y, xr.Dataset):
                y = y.to_array()
            y = y.data
            y = y.reshape(1, y.size)  # 1 observation with N dimensions
        else:
            raise NotImplementedError(
                f"infill_ds must be xr.Dataset or xr.DataArray, "
                f"not {type(infill_ds)}"
            )
        return y

    def _build_acquisition_fn_ds(self, X_s: Union[xr.Dataset, xr.DataArray]):
        """Initialise xr.DataArray for storing acquisition function values on search grid"""
        prepend_dims = ["iteration"]  # , "sample"]  # MC sample TODO
        prepend_coords = {
            "iteration": range(self.N_new_context),
            # "sample": range(self.n_samples_or_1),  # MC sample TODO
        }
        acquisition_fn_ds = create_empty_spatiotemporal_xarray(
            X=X_s,
            dates=self._get_times_from_tasks(),
            coord_names={"x1": self.x1_name, "x2": self.x2_name},
            data_vars=["acquisition_fn"],
            prepend_dims=prepend_dims,
            prepend_coords=prepend_coords,
        )["acquisition_fn"]
        acquisition_fn_ds.data[:] = np.nan

        return acquisition_fn_ds

    def _init_acquisition_fn_ds(self, X_s: xr.Dataset) -> xr.Dataset:
        """Instantiate acquisition function object"""
        # Unnormalise before instantiating
        X_s = self.model.data_processor.map_coords(X_s, unnorm=True)
        if isinstance(X_s, (xr.Dataset, xr.DataArray)):
            # xr.Dataset storing acquisition function values
            self.acquisition_fn_ds = self._build_acquisition_fn_ds(X_s)
        elif isinstance(X_s, (pd.DataFrame, pd.Series, pd.Index)):
            raise NotImplementedError(
                "Pandas support for active learning search points X_s not yet implemented."
            )
        else:
            raise TypeError(f"Unsupported type for X_s: {type(X_s)}")

    def _search(self, acquisition_fn: AcquisitionFunction):
        """
        Run one greedy pass by looping over each point in `X_s` and
        computing the acquisition function.

        If the search algorithm can be run over all points in parallel,
        this method should be overridden by the child class so that `self.run()`
        uses the parallel implementation.

        TODO check if below is still valid in GreedyOptimal:
        If the search method uses the y-values at search points (i.e. for
        an optimal benchmark), its `acquisition_fn` should expect a y_query input.
        """
        importances_list = []

        for task in self.tasks:
            # Parallel computation
            if isinstance(acquisition_fn, AcquisitionFunctionParallel):
                importances = acquisition_fn(task, self.X_s_arr)
                if self.pbar:
                    self.pbar.update(1)

            # Sequential computation
            elif isinstance(acquisition_fn, AcquisitionFunction):
                importances = []

                # TODO make computing the difference a bool
                # importance_bef = acquisition_fn(task)

                # Add size-1 dim after row dim to preserve row dim for passing to
                #   acquisition_fn. Also roll final axis to first axis for looping over search points.
                for x_query in np.rollaxis(self.X_s_arr[:, np.newaxis], 2):
                    y_query = self._sample_y_infill(
                        self.query_infill,
                        time=task["time"],
                        x1=x_query[0],
                        x2=x_query[1],
                    )
                    task_with_new = append_obs_to_task(
                        task, x_query, y_query, self.context_set_idx
                    )
                    # TODO this is a hack to add the auxiliary variable to the context set
                    if (
                        self.task_loader is not None
                        and self.task_loader.aux_at_contexts
                    ):
                        # Add auxiliary variable sampled at context set as a new context variable
                        X_c = task_with_new["X_c"][self.task_loader.aux_at_contexts[0]]
                        Y_c_aux = self.task_loader.sample_offgrid_aux(
                            X_c, self.task_loader.aux_at_contexts[1]
                        )
                        task_with_new["X_c"][-1] = X_c
                        task_with_new["Y_c"][-1] = Y_c_aux

                    importance = acquisition_fn(task_with_new)

                    # TODO make computing the difference a bool
                    # importance = importance - importance_bef

                    importances.append(importance)

                    if self.pbar:
                        self.pbar.update(1)

            else:
                allowed_classes = [
                    AcquisitionFunction,
                    AcquisitionFunctionParallel,
                    AcquisitionFunctionOracle,
                ]
                raise ValueError(
                    f"Acquisition function needs to inherit from one of {allowed_classes}."
                )

            importances = np.array(importances)
            importances_list.append(importances)

            if self.X_s_mask is not None:
                self.acquisition_fn_ds.loc[self.iteration, task["time"]].data[
                    self.X_s_mask.data
                ] = importances
            else:
                self.acquisition_fn_ds.loc[
                    self.iteration, task["time"]
                ] = importances.reshape(self.acquisition_fn_ds.shape[-2:])

        return np.mean(importances_list, axis=0)

    def _select_best(self, importances, X_s_arr):
        """Select sensor location corresponding to the best importance value.

        Appends the chosen search index to a list of chosen search indexes.
        """
        if self.min_or_max == "min":
            best_idx = np.argmin(importances)
        elif self.min_or_max == "max":
            best_idx = np.argmax(importances)

        best_x_query = X_s_arr[:, best_idx : best_idx + 1]

        # Index into original search space of chosen sensor location
        self.best_idxs_all.append(
            np.where((self.X_s_arr == best_x_query).all(axis=0))[0][0]
        )

        return best_x_query

    def _single_greedy_iteration(self, acquisition_fn: AcquisitionFunction):
        """
        Run a single greedy grid search iteration and append the optimal
        sensor location to self.X_new.
        """
        importances = self._search(acquisition_fn)
        best_x_query = self._select_best(importances, self.X_s_arr)

        self.X_new.append(best_x_query)

        return best_x_query

    def __call__(
        self,
        acquisition_fn: AcquisitionFunction,
        tasks: Union[List[Task], Task],
    ) -> Tuple[pd.DataFrame, xr.Dataset]:  # TODO is this correct use of typing?
        """
        Iteratively... docstring TODO

        Returns a tensor of proposed new sensor locations (in greedy iteration/priority order)
            and their corresponding list of indexes in the search space.
        """
        if (
            isinstance(acquisition_fn, AcquisitionFunctionOracle)
            and self.task_loader is None
        ):
            raise ValueError(
                "AcquisitionFunctionOracle requires a task_loader function to be passed to the GreedyOptimal constructor."
            )

        self.min_or_max = acquisition_fn.min_or_max
        if self.min_or_max not in ["min", "max"]:
            raise ValueError(
                f"min_or_max must be either 'min' or 'max', got {self.min_or_max}."
            )

        if isinstance(tasks, Task):
            tasks = [tasks]

        # Make deepcopys so that original tasks are not modified
        tasks = copy.deepcopy(tasks)

        # Add target set to tasks
        for i, task in enumerate(tasks):
            tasks[i]["X_t"][self.target_set_idx] = self.X_t_arr
            if isinstance(acquisition_fn, AcquisitionFunctionOracle):
                # Sample ground truth y-values at target points `self.X_t_arr` using `self.task_loader`
                date = tasks[i]["time"]
                task_with_Y_t = self.task_loader(
                    date, context_sampling=0, target_sampling=self.X_t_arr
                )
                tasks[i]["Y_t"] = task_with_Y_t["Y_t"]

            if "Y_t_aux" in tasks[i] and self.task_loader is None:
                raise ValueError(
                    "Model expects Y_t_aux data but a TaskLoader isn't provided to GreedyAlgorithm."
                )
            if self.task_loader is not None and self.task_loader.aux_at_target_dims > 0:
                tasks[i]["Y_t_aux"] = self.task_loader.sample_offgrid_aux(
                    self.X_t_arr, self.task_loader.aux_at_targets
                )

        self.tasks = tasks

        # Generate infill values at search points if not overridden
        if self.query_infill is None or self.proposed_infill is None:
            model_infill = self._model_infill_at_search_points(self.X_s)
            if self.query_infill is None:
                self.query_infill = model_infill
            if self.proposed_infill is None:
                self.proposed_infill = model_infill

        # Instantiate empty acquisition function object
        self._init_acquisition_fn_ds(self.X_s)

        # Dataframe for storing proposed context locations
        self.X_new_df = pd.DataFrame(columns=["x1", "x2"])
        self.X_new_df.index.name = "iteration"

        # List to track indexes into original search grid of chosen sensor locations
        #   as optimisation progresses. Used for filling y-values at chosen
        #   sensor locations, `self.X_new`
        self.best_idxs_all = []

        # Total iterations are number of new context points * number of tasks * number of search
        #   points (if not parallel) * number of Monte Carlo samples (if using MC)
        total_iterations = self.N_new_context * len(self.tasks)
        if not isinstance(acquisition_fn, AcquisitionFunctionParallel):
            total_iterations *= self.X_s_arr.shape[-1]
        # TODO make class attribute for list of sample methods
        if self.model_infill_method == "sample":
            total_iterations *= self.n_samples

        with tqdm(total=total_iterations, disable=not self.progress_bar) as self.pbar:
            for iteration in range(self.N_new_context):
                self.iteration = iteration
                x_new = self._single_greedy_iteration(acquisition_fn)

                # Append new proposed context points to each task
                for i, task in enumerate(self.tasks):
                    y_new = self._sample_y_infill(
                        self.proposed_infill,
                        time=task["time"],
                        x1=x_new[0],
                        x2=x_new[1],
                    )
                    self.tasks[i] = append_obs_to_task(
                        task, x_new, y_new, self.context_set_idx
                    )

                # Append new proposed context points to dataframe
                x_new_unnorm = self.model.data_processor.map_coord_array(
                    x_new, unnorm=True
                )
                self.X_new_df.loc[self.iteration] = x_new_unnorm.ravel()

        return self.X_new_df, self.acquisition_fn_ds
