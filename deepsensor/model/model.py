from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import (
    DataProcessor,
    process_X_mask_for_X,
    xarray_to_coord_array_normalised,
    mask_coord_array_normalised,
)
from deepsensor.model.pred import (
    Prediction,
    increase_spatial_resolution,
    infer_prediction_modality_from_X_t,
)
from deepsensor.data.task import Task

from typing import List, Union, Optional, Tuple
import copy

import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import lab as B

# For dispatching with TF and PyTorch model types when they have not yet been loaded.
# See https://beartype.github.io/plum/types.html#moduletype


class ProbabilisticModel:
    """
    Base class for probabilistic model used for DeepSensor.
    Ensures a set of methods required for DeepSensor
    are implemented by specific model classes that inherit from it.
    """

    def mean(self, task: Task, *args, **kwargs):
        """
        Computes the model mean prediction over target points based on given context
        data.

        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            :class:`numpy:numpy.ndarray`: Mean prediction over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def variance(self, task: Task, *args, **kwargs):
        """
        Model marginal variance over target points given context points.
        Shape (N,).

        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            :class:`numpy:numpy.ndarray`: Marginal variance over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def std(self, task: Task):
        """
        Model marginal standard deviation over target points given context
        points. Shape (N,).

        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            :class:`numpy:numpy.ndarray`: Marginal standard deviation over target points.
        """
        var = self.variance(task)
        return var**0.5

    def stddev(self, *args, **kwargs):
        return self.std(*args, **kwargs)

    def covariance(self, task: Task, *args, **kwargs):
        """
        Computes the model covariance matrix over target points based on given
        context data. Shape (N, N).

        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            :class:`numpy:numpy.ndarray`: Covariance matrix over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def mean_marginal_entropy(self, task: Task, *args, **kwargs):
        """
        Computes the mean marginal entropy over target points based on given
        context data.

        .. note::
            Note: Getting a vector of marginal entropies would be useful too.


        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            float: Mean marginal entropy over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def joint_entropy(self, task: Task, *args, **kwargs):
        """
        Computes the model joint entropy over target points based on given
        context data.


        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            float: Joint entropy over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def logpdf(self, task: Task, *args, **kwargs):
        """
        Computes the joint model logpdf over target points based on given
        context data.

        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            float: Joint logpdf over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def loss(self, task: Task, *args, **kwargs):
        """
        Computes the model loss over target points based on given context data.

        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.

        Returns:
            float: Loss over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()

    def sample(self, task: Task, n_samples=1, *args, **kwargs):
        """
        Draws ``n_samples`` joint samples over target points based on given
        context data. Returned shape is ``(n_samples, n_target)``.


        Args:
            task (:class:`~.data.task.Task`):
                Task containing context data.
            n_samples (int, optional):
                Number of samples to draw. Defaults to 1.

        Returns:
            tuple[:class:`numpy:numpy.ndarray`]: Joint samples over target points.

        Raises:
            NotImplementedError
                If not implemented by child class.
        """
        raise NotImplementedError()


class DeepSensorModel(ProbabilisticModel):
    """
    Implements DeepSensor prediction functionality of a ProbabilisticModel.
    Allows for outputting an xarray object containing on-grid predictions or a
    pandas object containing off-grid predictions.

    Args:
        data_processor (:class:`~.data.processor.DataProcessor`):
            DataProcessor object, used to unnormalise predictions.
        task_loader (:class:`~.data.loader.TaskLoader`):
            TaskLoader object, used to determine target variables for unnormalising.
    """

    N_mixture_components = 1  # Number of mixture components for mixture likelihoods

    def __init__(
        self,
        data_processor: Optional[DataProcessor] = None,
        task_loader: Optional[TaskLoader] = None,
    ):
        self.task_loader = task_loader
        self.data_processor = data_processor

    def predict(
        self,
        tasks: Union[List[Task], Task],
        X_t: Union[
            xr.Dataset,
            xr.DataArray,
            pd.DataFrame,
            pd.Series,
            pd.Index,
            np.ndarray,
        ],
        X_t_mask: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        X_t_is_normalised: bool = False,
        aux_at_targets_override: Union[xr.Dataset, xr.DataArray] = None,
        aux_at_targets_override_is_normalised: bool = False,
        resolution_factor: int = 1,
        pred_params: tuple[str] = ("mean", "std"),
        n_samples: int = 0,
        ar_sample: bool = False,
        ar_subsample_factor: int = 1,
        unnormalise: bool = True,
        seed: int = 0,
        append_indexes: dict = None,
        progress_bar: int = 0,
        verbose: bool = False,
    ) -> Prediction:
        """
        Predict on a regular grid or at off-grid locations.

        Args:
            tasks (List[Task] | Task):
                List of tasks containing context data.
            X_t (:class:`xarray.Dataset` | :class:`xarray.DataArray` | :class:`pandas.DataFrame` | :class:`pandas.Series` | :class:`pandas.Index` | :class:`numpy:numpy.ndarray`):
                Target locations to predict at. Can be an xarray object
                containingon-grid locations or a pandas object containing off-grid locations.
            X_t_mask: :class:`xarray.Dataset` | :class:`xarray.DataArray`, optional
                2D mask to apply to gridded ``X_t`` (zero/False will be NaNs). Will be interpolated
                to the same grid as ``X_t``. Default None (no mask).
            X_t_is_normalised (bool):
                Whether the ``X_t`` coords are normalised. If False, will normalise
                the coords before passing to model. Default ``False``.
            aux_at_targets_override (:class:`xarray.Dataset` | :class:`xarray.DataArray`):
                Optional auxiliary xarray data to override from the task_loader.
            aux_at_targets_override_is_normalised (bool):
                Whether the `aux_at_targets_override` coords are normalised.
                If False, the DataProcessor will normalise the coords before passing to model.
                Default False.
            pred_params (tuple[str]):
                Tuple of prediction parameters to return. The strings refer to methods
                of the model class which will be called and stored in the Prediction object.
                Default ("mean", "std").
            resolution_factor (float):
                Optional factor to increase the resolution of the target grid
                by. E.g. 2 will double the target resolution, 0.5 will halve
                it.Applies to on-grid predictions only. Default 1.
            n_samples (int):
                Number of joint samples to draw from the model. If 0, will not
                draw samples. Default 0.
            ar_sample (bool):
                Whether to use autoregressive sampling. Default ``False``.
            unnormalise (bool):
                Whether to unnormalise the predictions. Only works if ``self``
                hasa ``data_processor`` and ``task_loader`` attribute. Default
                ``True``.
            seed (int):
                Random seed for deterministic sampling. Default 0.
            append_indexes (dict):
                Dictionary of index metadata to append to pandas indexes in the
                off-grid case. Default ``None``.
            progress_bar (int):
                Whether to display a progress bar over tasks. Default 0.
            verbose (bool):
                Whether to print time taken for prediction. Default ``False``.

        Returns:
            :class:`~.model.pred.Prediction`):
                A `dict`-like object mapping from target variable IDs to xarray or pandas objects
                containing model predictions.
                - If ``X_t`` is a pandas object, returns pandas objects
                containing off-grid predictions.
                - If ``X_t`` is an xarray object, returns xarray object
                containing on-grid predictions.
                - If ``n_samples`` == 0, returns only mean and std predictions.
                - If ``n_samples`` > 0, returns mean, std and samples
                predictions.

        Raises:
            ValueError
                If ``X_t`` is not an xarray object and
                ``resolution_factor`` is not 1 or ``ar_subsample_factor`` is
                not 1.
            ValueError
                If ``X_t`` is not a pandas object and ``append_indexes`` is not
                ``None``.
            ValueError
                If ``X_t`` is not an xarray, pandas or numpy object.
            ValueError
                If ``append_indexes`` are not all the same length as ``X_t``.
        """
        tic = time.time()
        mode = infer_prediction_modality_from_X_t(X_t)
        if not isinstance(X_t, (xr.DataArray, xr.Dataset)):
            if resolution_factor != 1:
                raise ValueError(
                    "resolution_factor can only be used with on-grid predictions."
                )
            if ar_subsample_factor != 1:
                raise ValueError(
                    "ar_subsample_factor can only be used with on-grid predictions."
                )
        if not isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index, np.ndarray)):
            if append_indexes is not None:
                raise ValueError(
                    "append_indexes can only be used with off-grid predictions."
                )
        if mode == "off-grid" and X_t_mask is not None:
            # TODO: Unit test this
            raise ValueError("X_t_mask can only be used with on-grid predictions.")
        if ar_sample and n_samples < 1:
            raise ValueError("Must pass `n_samples` > 0 to use `ar_sample`.")

        if type(tasks) is Task:
            tasks = [tasks]

        if n_samples >= 1:
            B.set_random_seed(seed)
            np.random.seed(seed)

        dates = [task["time"] for task in tasks]

        # Flatten tuple of tuples to single list
        target_var_IDs = [
            var_ID for set in self.task_loader.target_var_IDs for var_ID in set
        ]

        # TODO consider removing this logic, can we just depend on the dim names in X_t?
        if not unnormalise:
            coord_names = {"x1": "x1", "x2": "x2"}
        elif unnormalise:
            coord_names = {
                "x1": self.data_processor.raw_spatial_coord_names[0],
                "x2": self.data_processor.raw_spatial_coord_names[1],
            }

        ### Pre-process X_t if necessary (TODO consider moving this to Prediction class)
        if isinstance(X_t, pd.Index):
            X_t = pd.DataFrame(index=X_t)
        elif isinstance(X_t, np.ndarray):
            # Convert to empty dataframe with normalised or unnormalised coord names
            if X_t_is_normalised:
                index_names = ["x1", "x2"]
            else:
                index_names = self.data_processor.raw_spatial_coord_names
            X_t = pd.DataFrame(X_t.T, columns=index_names)
            X_t = X_t.set_index(index_names)
        elif isinstance(X_t, (xr.DataArray, xr.Dataset)):
            # Remove time dimension if present
            if "time" in X_t.coords:
                X_t = X_t.isel(time=0).drop("time")

        if mode == "off-grid" and append_indexes is not None:
            # Check append_indexes are all same length as X_t
            if append_indexes is not None:
                for idx, vals in append_indexes.items():
                    if len(vals) != len(X_t):
                        raise ValueError(
                            f"append_indexes[{idx}] must be same length as X_t, got {len(vals)} and {len(X_t)} respectively."
                        )
            X_t = X_t.reset_index()
            X_t = pd.concat([X_t, pd.DataFrame(append_indexes)], axis=1)
            X_t = X_t.set_index(list(X_t.columns))

        if X_t_is_normalised:
            X_t_normalised = X_t
            # Unnormalise coords to use for xarray/pandas objects for storing predictions
            X_t = self.data_processor.map_coords(X_t, unnorm=True)
        elif not X_t_is_normalised:
            # Normalise coords to use for model
            X_t_normalised = self.data_processor.map_coords(X_t)

        if mode == "on-grid":
            if resolution_factor != 1:
                X_t_normalised = increase_spatial_resolution(
                    X_t_normalised, resolution_factor
                )
                X_t = increase_spatial_resolution(
                    X_t, resolution_factor, coord_names=coord_names
                )
            if X_t_mask is not None:
                X_t_mask = process_X_mask_for_X(X_t_mask, X_t)
                X_t_mask_normalised = self.data_processor.map_coords(X_t_mask)
                X_t_arr = xarray_to_coord_array_normalised(X_t_normalised)
                # Remove points that lie outside the mask
                X_t_arr = mask_coord_array_normalised(X_t_arr, X_t_mask_normalised)
            else:
                X_t_arr = (
                    X_t_normalised["x1"].values,
                    X_t_normalised["x2"].values,
                )
        elif mode == "off-grid":
            X_t_arr = X_t_normalised.reset_index()[["x1", "x2"]].values.T

        if isinstance(X_t_arr, tuple):
            target_shape = (len(X_t_arr[0]), len(X_t_arr[1]))
        else:
            target_shape = (X_t_arr.shape[1],)

        if not unnormalise:
            X_t = X_t_normalised

        if "mixture_probs" in pred_params:
            # Store each mixture component separately w/o overriding pred_params
            pred_params_to_store = copy.deepcopy(pred_params)
            pred_params_to_store.remove("mixture_probs")
            for component_i in range(self.N_mixture_components):
                pred_params_to_store.append(f"mixture_probs_{component_i}")
        else:
            pred_params_to_store = pred_params

        # Dict to store predictions for each target variable
        pred = Prediction(
            target_var_IDs,
            pred_params_to_store,
            dates,
            X_t,
            X_t_mask,
            coord_names,
            n_samples=n_samples,
        )

        def unnormalise_pred_array(arr, **kwargs):
            """Unnormalise an (N_dims, N_targets) array of predictions."""
            var_IDs_flattened = [
                var_ID
                for var_IDs in self.task_loader.target_var_IDs
                for var_ID in var_IDs
            ]
            assert arr.shape[0] == len(
                var_IDs_flattened
            ), f"{arr.shape[0]} != {len(var_IDs_flattened)}"
            for i, var_ID in enumerate(var_IDs_flattened):
                arr[i] = self.data_processor.map_array(
                    arr[i],
                    var_ID,
                    method=self.data_processor.config[var_ID]["method"],
                    unnorm=True,
                    **kwargs,
                )
            return arr

        # Don't change tasks by reference when overriding target locations
        # TODO consider not copying tasks by default for efficiency
        tasks = copy.deepcopy(tasks)

        if self.task_loader.aux_at_targets is not None:
            if aux_at_targets_override is not None:
                aux_at_targets = aux_at_targets_override
                if not aux_at_targets_override_is_normalised:
                    # Assumes using default normalisation method
                    aux_at_targets = self.data_processor(
                        aux_at_targets, assert_computed=True
                    )
            else:
                aux_at_targets = self.task_loader.aux_at_targets

        for task in tqdm(tasks, position=0, disable=progress_bar < 1, leave=True):
            task["X_t"] = [X_t_arr for _ in range(len(self.task_loader.target_var_IDs))]

            # If passing auxiliary data, need to sample it at target locations
            if self.task_loader.aux_at_targets is not None:
                aux_at_targets_sliced = self.task_loader.time_slice_variable(
                    aux_at_targets, task["time"]
                )
                task["Y_t_aux"] = self.task_loader.sample_offgrid_aux(
                    X_t_arr, aux_at_targets_sliced
                )

            prediction_arrs = {}
            prediction_methods = {}
            for param in pred_params:
                try:
                    method = getattr(self, param)
                    prediction_methods[param] = method
                except AttributeError:
                    raise AttributeError(
                        f"Prediction method {param} not found in model class."
                    )
            if n_samples >= 1:
                B.set_random_seed(seed)
                np.random.seed(seed)
                if ar_sample:
                    sample_method = getattr(self, "ar_sample")
                    sample_args = {
                        "n_samples": n_samples,
                        "ar_subsample_factor": ar_subsample_factor,
                    }
                else:
                    sample_method = getattr(self, "sample")
                    sample_args = {"n_samples": n_samples}

            # If `DeepSensor` model child has been sub-classed with a `__call__` method,
            # we assume this is a distribution-like object that can be used to compute
            # mean, std and samples. Otherwise, run the model with `Task` for each prediction type.
            if hasattr(self, "__call__"):
                # Run model forwards once to generate output distribution, which we re-use
                dist = self(task, n_samples=n_samples)
                for param, method in prediction_methods.items():
                    prediction_arrs[param] = method(dist)
                if n_samples >= 1 and not ar_sample:
                    samples_arr = sample_method(dist, **sample_args)
                    # samples_arr = samples_arr.reshape((n_samples, len(target_var_IDs), *target_shape))
                    prediction_arrs["samples"] = samples_arr
                elif n_samples >= 1 and ar_sample:
                    # Can't draw AR samples from distribution object, need to re-run with task
                    samples_arr = sample_method(task, **sample_args)
                    samples_arr = samples_arr.reshape(
                        (n_samples, len(target_var_IDs), *target_shape)
                    )
                    prediction_arrs["samples"] = samples_arr
            else:
                # Re-run model for each prediction type
                for param, method in prediction_methods.items():
                    prediction_arrs[param] = method(task)
                if n_samples >= 1:
                    samples_arr = sample_method(task, **sample_args)
                    if ar_sample:
                        samples_arr = samples_arr.reshape(
                            (n_samples, len(target_var_IDs), *target_shape)
                        )
                    prediction_arrs["samples"] = samples_arr

            # Concatenate multi-target predictions
            for param, arr in prediction_arrs.items():
                if isinstance(arr, (list, tuple)):
                    if param != "samples":
                        concat_axis = 0
                    elif param == "samples":
                        # Axis 0 is sample dim, axis 1 is variable dim
                        concat_axis = 1
                    prediction_arrs[param] = np.concatenate(arr, axis=concat_axis)

            # Unnormalise predictions
            for param, arr in prediction_arrs.items():
                # TODO make class attributes?
                scale_and_offset_params = ["mean"]
                scale_only_params = ["std"]
                scale_squared_only_params = ["variance"]
                if unnormalise:
                    if param == "samples":
                        for sample_i in range(n_samples):
                            prediction_arrs["samples"][sample_i] = (
                                unnormalise_pred_array(
                                    prediction_arrs["samples"][sample_i]
                                )
                            )
                    elif param in scale_and_offset_params:
                        prediction_arrs[param] = unnormalise_pred_array(arr)
                    elif param in scale_only_params:
                        prediction_arrs[param] = unnormalise_pred_array(
                            arr, add_offset=False
                        )
                    elif param in scale_squared_only_params:
                        # This is a horrible hack to repeat the scaling operation of the linear
                        #   transform twice s.t. new_var = scale ^ 2 * var
                        prediction_arrs[param] = unnormalise_pred_array(
                            arr, add_offset=False
                        )
                        prediction_arrs[param] = unnormalise_pred_array(
                            prediction_arrs[param], add_offset=False
                        )
                    else:
                        # Assume prediction parameters not captured above are dimensionless
                        #   quantities like probabilities and should not be unnormalised
                        pass

            # Assign predictions to Prediction object
            for param, arr in prediction_arrs.items():
                if param != "mixture_probs":
                    pred.assign(param, task["time"], arr)
                elif param == "mixture_probs":
                    assert arr.shape[0] == self.N_mixture_components, (
                        f"Number of mixture components ({arr.shape[0]}) does not match "
                        f"model attribute N_mixture_components ({self.N_mixture_components})."
                    )
                    for component_i, probs in enumerate(arr):
                        pred.assign(f"{param}_{component_i}", task["time"], probs)

        if verbose:
            dur = time.time() - tic
            print(f"Done in {np.floor(dur / 60)}m:{dur % 60:.0f}s.\n")

        return pred

    def predict_patch(
        self,
        tasks: Union[List[Task], Task],
        X_t: Union[
            xr.Dataset,
            xr.DataArray,
            pd.DataFrame,
            pd.Series,
            pd.Index,
            np.ndarray,
        ],
        data_processor: Union[
            xr.DataArray,
            xr.Dataset,
            pd.DataFrame,
            List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]],
        ],
        stride: Union[float, tuple[float]],
        patch_size: Union[float, tuple[float]],
        X_t_mask: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        X_t_is_normalised: bool = False,
        aux_at_targets_override: Union[xr.Dataset, xr.DataArray] = None,
        aux_at_targets_override_is_normalised: bool = False,
        resolution_factor: int = 1,
        pred_params: tuple[str] = ("mean", "std"),
        n_samples: int = 0,
        ar_sample: bool = False,
        ar_subsample_factor: int = 1,
        unnormalise: bool = False,
        seed: int = 0,
        append_indexes: dict = None,
        progress_bar: int = 0,
        verbose: bool = False,
    ) -> Prediction:
        """
        Predict on a regular grid or at off-grid locations.

        Args:
            tasks (List[Task] | Task):
                List of tasks containing context data.
            data_processor (:class:`~.data.processor.DataProcessor`):
                Used for unnormalising the coordinates of the bounding boxes of patches.
            stride (Union[float, tuple[float]]):
                Length of stride between adjacent patches in x1/x2 normalised coordinates. If passed a single float, will use value for both x1 & x2.
            patch_size (Union[float, tuple[float]]):
                Height and width of patch in x1/x2 normalised coordinates. If passed a single float, will use value for both x1 & x2.
            X_t (:class:`xarray.Dataset` | :class:`xarray.DataArray` | :class:`pandas.DataFrame` | :class:`pandas.Series` | :class:`pandas.Index` | :class:`numpy:numpy.ndarray`):
                Target locations to predict at. Can be an xarray object
                containingon-grid locations or a pandas object containing off-grid locations.
            X_t_mask: :class:`xarray.Dataset` | :class:`xarray.DataArray`, optional
                2D mask to apply to gridded ``X_t`` (zero/False will be NaNs). Will be interpolated
                to the same grid as ``X_t``. Default None (no mask).
            X_t_is_normalised (bool):
                Whether the ``X_t`` coords are normalised. If False, will normalise
                the coords before passing to model. Default ``False``.
            aux_at_targets_override (:class:`xarray.Dataset` | :class:`xarray.DataArray`):
                Optional auxiliary xarray data to override from the task_loader.
            aux_at_targets_override_is_normalised (bool):
                Whether the `aux_at_targets_override` coords are normalised.
                If False, the DataProcessor will normalise the coords before passing to model.
                Default False.
            pred_params (tuple[str]):
                Tuple of prediction parameters to return. The strings refer to methods
                of the model class which will be called and stored in the Prediction object.
                Default ("mean", "std").
            resolution_factor (float):
                Optional factor to increase the resolution of the target grid
                by. E.g. 2 will double the target resolution, 0.5 will halve
                it.Applies to on-grid predictions only. Default 1.
            n_samples (int):
                Number of joint samples to draw from the model. If 0, will not
                draw samples. Default 0.
            ar_sample (bool):
                Whether to use autoregressive sampling. Default ``False``.
            unnormalise (bool):
                Whether to unnormalise the predictions. Only works if ``self``
                hasa ``data_processor`` and ``task_loader`` attribute. Default
                ``True``.
            seed (int):
                Random seed for deterministic sampling. Default 0.
            append_indexes (dict):
                Dictionary of index metadata to append to pandas indexes in the
                off-grid case. Default ``None``.
            progress_bar (int):
                Whether to display a progress bar over tasks. Default 0.
            verbose (bool):
                Whether to print time taken for prediction. Default ``False``.

        Returns:
            :class:`~.model.pred.Prediction`):
                A `dict`-like object mapping from target variable IDs to xarray or pandas objects
                containing model predictions.
                - If ``X_t`` is a pandas object, returns pandas objects
                containing off-grid predictions.
                - If ``X_t`` is an xarray object, returns xarray object
                containing on-grid predictions.
                - If ``n_samples`` == 0, returns only mean and std predictions.
                - If ``n_samples`` > 0, returns mean, std and samples
                predictions.

        Raises:
            ValueError
                If ``X_t`` is not an xarray object and
                ``resolution_factor`` is not 1 or ``ar_subsample_factor`` is
                not 1.
            ValueError
                If ``X_t`` is not a pandas object and ``append_indexes`` is not
                ``None``.
            ValueError
                If ``X_t`` is not an xarray, pandas or numpy object.
            ValueError
                If ``append_indexes`` are not all the same length as ``X_t``.
        """

        def get_patches_per_row(preds) -> int:
            """
            Calculate number of patches per row.
            Required to stitch patches back together.
            Args:
                preds (List[class:`~.model.pred.Prediction`]):
                        A list of `dict`-like objects containing patchwise predictions.

            Returns:
                patches_per_row: int
                    Number of patches per row.
            """
            patches_per_row = 0
            vars = list(preds[0][0].data_vars)
            var = vars[0]
            y_val = preds[0][0][var].coords[unnorm_coord_names["x1"]].min()

            for p in preds:
                if p[0][var].coords[unnorm_coord_names["x1"]].min() == y_val:
                    patches_per_row = patches_per_row + 1

            return patches_per_row

        def get_patch_overlap(overlap_norm, data_processor, X_t_ds) -> int:
            """
            Calculate overlap between adjacent patches in pixels.

            Parameters
            ----------
            overlap_norm : tuple[float].
                Normalised size of overlap in x1/x2.

            data_processor (:class:`~.data.processor.DataProcessor`):
                Used for unnormalising the coordinates of the bounding boxes of patches.

            X_t_ds (:class:`xarray.Dataset` | :class:`xarray.DataArray` | :class:`pandas.DataFrame` | :class:`pandas.Series` | :class:`pandas.Index` | :class:`numpy:numpy.ndarray`):
                Data array containing target locations to predict at.

            Returns
            -------
            patch_overlap : tuple (int)
                Unnormalised size of overlap between adjacent patches.
            """
            # Place stride and patch size values in Xarray to pass into unnormalise()
            overlap_list = [0, overlap_norm[0], 0, overlap_norm[1]]
            x1 = xr.DataArray([overlap_list[0], overlap_list[1]], dims="x1", name="x1")
            x2 = xr.DataArray([overlap_list[2], overlap_list[3]], dims="x2", name="x2")
            overlap_norm_xr = xr.Dataset(coords={"x1": x1, "x2": x2})

            # Unnormalise coordinates of bounding boxes
            overlap_unnorm_xr = data_processor.unnormalise(overlap_norm_xr)
            unnorm_overlap_x1 = overlap_unnorm_xr.coords[
                unnorm_coord_names["x1"]
            ].values[1]
            unnorm_overlap_x2 = overlap_unnorm_xr.coords[
                unnorm_coord_names["x2"]
            ].values[1]

            # Find the position of these indices within the DataArray
            x_overlap_index = int(
                np.ceil(
                    (
                        np.argmin(
                            np.abs(
                                X_t_ds.coords[unnorm_coord_names["x1"]].values
                                - unnorm_overlap_x1
                            )
                        )
                        / 2
                    )
                )
            )
            y_overlap_index = int(
                np.ceil(
                    (
                        np.argmin(
                            np.abs(
                                X_t_ds.coords[unnorm_coord_names["x2"]].values
                                - unnorm_overlap_x2
                            )
                        )
                        / 2
                    )
                )
            )
            xy_overlap = (x_overlap_index, y_overlap_index)

            return xy_overlap

        def get_index(*args, x1=True) -> Union[int, Tuple[List[int], List[int]]]:
            """
            Convert coordinates into pixel row/column (index).

            Parameters
            ----------
            args : tuple
                If one argument (numeric), it represents the coordinate value.
                If two arguments (lists), they represent lists of coordinate values.

            x1 : bool, optional
                If True, compute index for x1 (default is True).

            Returns
            -------
                Union[int, Tuple[List[int], List[int]]]
                If one argument is provided and x1 is True or False, returns the index position.
                If two arguments are provided, returns a tuple containing two lists:
                - First list: indices corresponding to x1 coordinates.
                - Second list: indices corresponding to x2 coordinates.

            """
            if len(args) == 1:
                patch_coord = args
                if x1:
                    coord_index = np.argmin(
                        np.abs(
                            X_t.coords[unnorm_coord_names["x1"]].values - patch_coord
                        )
                    )
                else:
                    coord_index = np.argmin(
                        np.abs(
                            X_t.coords[unnorm_coord_names["x2"]].values - patch_coord
                        )
                    )
                return coord_index

            elif len(args) == 2:
                patch_x1, patch_x2 = args
                x1_index = [
                    np.argmin(
                        np.abs(X_t.coords[unnorm_coord_names["x1"]].values - target_x1)
                    )
                    for target_x1 in patch_x1
                ]
                x2_index = [
                    np.argmin(
                        np.abs(X_t.coords[unnorm_coord_names["x2"]].values - target_x2)
                    )
                    for target_x2 in patch_x2
                ]
                return (x1_index, x2_index)

        def stitch_clipped_predictions(
            patch_preds, patch_overlap, patches_per_row
        ) -> dict:
            """
            Stitch patchwise predictions to form prediction at original extent.

            Parameters
            ----------
            patch_preds : list (class:`~.model.pred.Prediction`)
                List of patchwise predictions

            patch_overlap: int
                Overlap between adjacent patches in pixels.

            patches_per_row: int
                Number of patchwise predictions in each row.

            Returns
            -------
            combined: dict
                Dictionary object containing the stitched model predictions.
            """

            data_x1 = (
                X_t.coords[unnorm_coord_names["x1"]].min().values,
                X_t.coords[unnorm_coord_names["x1"]].max().values,
            )
            data_x2 = (
                X_t.coords[unnorm_coord_names["x2"]].min().values,
                X_t.coords[unnorm_coord_names["x2"]].max().values,
            )
            data_x1_index, data_x2_index = get_index(data_x1, data_x2)
            patches_clipped = {var_name: [] for var_name in patch_preds[0].keys()}

            for i, patch_pred in enumerate(patch_preds):
                for var_name, data_array in patch_pred.items():  # previously patch
                    if var_name in patch_pred:
                        # Get row/col index values of each patch
                        patch_x1 = (
                            data_array.coords[unnorm_coord_names["x1"]].min().values,
                            data_array.coords[unnorm_coord_names["x1"]].max().values,
                        )
                        patch_x2 = (
                            data_array.coords[unnorm_coord_names["x2"]].min().values,
                            data_array.coords[unnorm_coord_names["x2"]].max().values,
                        )
                        patch_x1_index, patch_x2_index = get_index(patch_x1, patch_x2)

                        b_x1_min, b_x1_max = patch_overlap[0], patch_overlap[0]
                        b_x2_min, b_x2_max = patch_overlap[1], patch_overlap[1]
                        # Do not remove border for the patches along top and left of dataset
                        # and change overlap size for last patch in each row and column.
                        if patch_x2_index[0] == data_x2_index[0]:
                            b_x2_min = 0
                        elif patch_x2_index[1] == data_x2_index[1]:
                            b_x2_max = 0
                            patch_row_prev = preds[i - 1]
                            prev_patch_x2_max = get_index(
                                int(
                                    patch_row_prev[var_name]
                                    .coords[unnorm_coord_names["x2"]]
                                    .max()
                                ),
                                x1=False,
                            )
                            b_x2_min = (
                                prev_patch_x2_max - patch_x2_index[0]
                            ) - patch_overlap[1]

                        if patch_x1_index[0] == data_x1_index[0]:
                            b_x1_min = 0
                        elif abs(patch_x1_index[1] - data_x1_index[1]) < 2:
                            b_x1_max = 0
                            patch_prev = preds[i - patches_per_row]
                            prev_patch_x1_max = get_index(
                                int(
                                    patch_prev[var_name]
                                    .coords[unnorm_coord_names["x1"]]
                                    .max()
                                ),
                                x1=True,
                            )
                            b_x1_min = (
                                prev_patch_x1_max - patch_x1_index[0]
                            ) - patch_overlap[0]

                        patch_clip_x1_min = int(b_x1_min)
                        patch_clip_x1_max = int(
                            data_array.sizes[unnorm_coord_names["x1"]] - b_x1_max
                        )
                        patch_clip_x2_min = int(b_x2_min)
                        patch_clip_x2_max = int(
                            data_array.sizes[unnorm_coord_names["x2"]] - b_x2_max
                        )

                        patch_clip = data_array[
                            {
                                unnorm_coord_names["x1"]: slice(
                                    patch_clip_x1_min, patch_clip_x1_max
                                ),
                                unnorm_coord_names["x2"]: slice(
                                    patch_clip_x2_min, patch_clip_x2_max
                                ),
                            }
                        ]

                        patches_clipped[var_name].append(patch_clip)

            combined = {
                var_name: xr.combine_by_coords(patches, compat="no_conflicts")
                for var_name, patches in patches_clipped.items()
            }

            return combined

        # sanitise patch_size and stride arguments

        if isinstance(patch_size, float) and patch_size is not None:
            patch_size = (patch_size, patch_size)

        if isinstance(stride, float) and stride is not None:
            stride = (stride, stride)

        if stride[0] > patch_size[0] or stride[1] > patch_size[1]:
            raise ValueError(
                f"stride must be smaller than patch_size in the corresponding dimensions. Got: patch_size: {patch_size}, stride: {stride}"
            )

        for val in list(stride + patch_size):
            if val > 1.0 or val < 0.0:
                raise ValueError(
                    f"Values of stride and patch_size must be between 0 & 1. Got: patch_size: {patch_size}, stride: {stride}"
                )

        # Get coordinate names of original unnormalised dataset.
        unnorm_coord_names = {
            "x1": self.data_processor.raw_spatial_coord_names[0],
            "x2": self.data_processor.raw_spatial_coord_names[1],
        }

        # tasks should be iterable, if only one is provided, make it a list
        if type(tasks) is Task:
            tasks = [tasks]

        # Perform patchwise predictions
        preds = []
        for task in tasks:
            bbox = task["bbox"]

            if bbox is None:
                raise AttributeError(
                    "For patchwise prediction, only tasks generated using a patch_strategy of 'sliding' are valid. \
                        This task has a bbox value of None, indicating that it was generated with a patch_strategy of \
                            'random' or None."
                )

            # Unnormalise coordinates of bounding box of patch
            x1 = xr.DataArray([bbox[0], bbox[1]], dims="x1", name="x1")
            x2 = xr.DataArray([bbox[2], bbox[3]], dims="x2", name="x2")
            bbox_norm = xr.Dataset(coords={"x1": x1, "x2": x2})
            bbox_unnorm = data_processor.unnormalise(bbox_norm)
            unnorm_bbox_x1 = (
                bbox_unnorm[unnorm_coord_names["x1"]].values.min(),
                bbox_unnorm[unnorm_coord_names["x1"]].values.max(),
            )
            unnorm_bbox_x2 = (
                bbox_unnorm[unnorm_coord_names["x2"]].values.min(),
                bbox_unnorm[unnorm_coord_names["x2"]].values.max(),
            )

            # Determine X_t for patch
            task_extent_dict = {
                unnorm_coord_names["x1"]: slice(unnorm_bbox_x1[0], unnorm_bbox_x1[1]),
                unnorm_coord_names["x2"]: slice(unnorm_bbox_x2[0], unnorm_bbox_x2[1]),
            }
            task_X_t = X_t.sel(**task_extent_dict)

            # Patchwise prediction
            pred = self.predict(task, task_X_t)
            # Append patchwise DeepSensor prediction object to list
            preds.append(pred)

        overlap_norm = tuple(
            patch - stride for patch, stride in zip(patch_size, stride)
        )
        patch_overlap_unnorm = get_patch_overlap(overlap_norm, data_processor, X_t)
        patches_per_row = get_patches_per_row(preds)
        stitched_prediction = stitch_clipped_predictions(
            preds, patch_overlap_unnorm, patches_per_row
        )

        ## Cast prediction into DeepSensor.Prediction object.
        # TODO make this into seperate method.
        prediction = copy.deepcopy(preds[0])

        # Generate new blank DeepSensor.prediction object in original coordinate system.
        for var_name_copy, data_array_copy in prediction.items():

            # set x and y coords
            stitched_preds = xr.Dataset(
                coords={
                    "x1": X_t[unnorm_coord_names["x1"]],
                    "x2": X_t[unnorm_coord_names["x2"]],
                }
            )

            # Set time to same as patched prediction
            stitched_preds["time"] = data_array_copy["time"]

            # set variable names to those in patched prediction, make values blank
            for var_name_i in data_array_copy.data_vars:
                stitched_preds[var_name_i] = data_array_copy[var_name_i]
                stitched_preds[var_name_i][:] = np.nan
            prediction[var_name_copy] = stitched_preds
            prediction[var_name_copy] = stitched_prediction[var_name_copy]

        return prediction


def main():  # pragma: no cover
    import deepsensor.tensorflow
    from deepsensor.data.loader import TaskLoader
    from deepsensor.data.processor import DataProcessor
    from deepsensor.model.convnp import ConvNP

    import xarray as xr
    import pandas as pd
    import numpy as np

    # Load raw data
    ds_raw = xr.tutorial.open_dataset("air_temperature")["air"]
    ds_raw2 = copy.deepcopy(ds_raw)
    ds_raw2.name = "air2"

    # Normalise data
    data_processor = DataProcessor(x1_name="lat", x2_name="lon")
    ds = data_processor(ds_raw)
    ds2 = data_processor(ds_raw2)

    # Set up task loader
    task_loader = TaskLoader(context=ds, target=[ds, ds2])

    # Set up model
    model = ConvNP(data_processor, task_loader)

    # Predict on new task with 10% of context data and a dense grid of target points
    test_tasks = task_loader(
        pd.date_range("2014-12-25", "2014-12-31"), context_sampling=40
    )
    # print(repr(test_tasks))

    X_t = ds_raw
    pred = model.predict(test_tasks, X_t=X_t, n_samples=5)
    print(pred)

    X_t = np.zeros((2, 1))
    pred = model.predict(test_tasks, X_t=X_t, X_t_is_normalised=True)
    print(pred)

    # DEBUG
    # task = task_loader("2014-12-31", context_sampling=40, target_sampling="all")
    # samples = model.ar_sample(task, 5, ar_subsample_factor=20)


if __name__ == "__main__":  # pragma: no cover
    main()
