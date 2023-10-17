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


def create_empty_spatiotemporal_xarray(
    X: Union[xr.Dataset, xr.DataArray],
    dates: List,
    coord_names: dict = {"x1": "x1", "x2": "x2"},
    data_vars: List[str] = ["var"],
    prepend_dims: Optional[List[str]] = None,
    prepend_coords: Optional[dict] = None,
):
    """
    ...

    Args:
        X (:class:`xarray.Dataset` | :class:`xarray.DataArray`):
            ...
        dates (List[...]):
            ...
        coord_names (dict, optional):
            ..., by default {"x1": "x1", "x2": "x2"}
        data_vars (List[str], optional):
            ..., by default ["var"]
        prepend_dims (List[str], optional):
            ..., by default None
        prepend_coords (dict, optional):
            ..., by default None

    Returns:
        ...
            ...

    Raises:
        ValueError
            If ``data_vars`` contains duplicate values.
        ValueError
            If ``coord_names["x1"]`` is not uniformly spaced.
        ValueError
            If ``coord_names["x2"]`` is not uniformly spaced.
        ValueError
            If ``prepend_dims`` and ``prepend_coords`` are not the same length.
    """
    if prepend_dims is None:
        prepend_dims = []
    if prepend_coords is None:
        prepend_coords = {}

    # Check for any repeated data_vars
    if len(data_vars) != len(set(data_vars)):
        raise ValueError(
            f"Duplicate data_vars found in data_vars: {data_vars}. "
            "This would cause the xarray.Dataset to have fewer variables than expected."
        )

    x1_predict = X.coords[coord_names["x1"]]
    x2_predict = X.coords[coord_names["x2"]]

    # Assert uniform spacing
    if not np.allclose(np.diff(x1_predict), np.diff(x1_predict)[0]):
        raise ValueError(f"Coordinate {coord_names['x1']} must be uniformly spaced.")
    if not np.allclose(np.diff(x2_predict), np.diff(x2_predict)[0]):
        raise ValueError(f"Coordinate {coord_names['x2']} must be uniformly spaced.")

    if len(prepend_dims) != len(set(prepend_dims)):
        # TODO unit test
        raise ValueError(
            f"Length of prepend_dims ({len(prepend_dims)}) must be equal to length of "
            f"prepend_coords ({len(prepend_coords)})."
        )

    dims = [*prepend_dims, "time", coord_names["x1"], coord_names["x2"]]
    coords = {
        **prepend_coords,
        "time": pd.to_datetime(dates),
        coord_names["x1"]: x1_predict,
        coord_names["x2"]: x2_predict,
    }

    pred_ds = xr.Dataset(
        {data_var: xr.DataArray(dims=dims, coords=coords) for data_var in data_vars}
    ).astype("float32")

    # Convert time coord to pandas timestamps
    pred_ds = pred_ds.assign_coords(time=pd.to_datetime(pred_ds.time.values))

    # TODO: Convert init time to forecast time?
    # pred_ds = pred_ds.assign_coords(
    #     time=pred_ds['time'] + pd.Timedelta(days=task_loader.target_delta_t[0]))

    return pred_ds


def increase_spatial_resolution(
    X_t_normalised,
    resolution_factor,
    coord_names: dict = {"x1": "x1", "x2": "x2"},
):
    """
    ...

    ..
        # TODO wasteful to interpolate X_t_normalised

    Args:
        X_t_normalised (...):
            ...
        resolution_factor (...):
            ...
        coord_names (dict, optional):
            ..., by default {"x1": "x1", "x2": "x2"}

    Returns:
        ...
            ...
    """
    assert isinstance(resolution_factor, (float, int))
    assert isinstance(X_t_normalised, (xr.DataArray, xr.Dataset))
    x1_name, x2_name = coord_names["x1"], coord_names["x2"]
    x1, x2 = X_t_normalised.coords[x1_name], X_t_normalised.coords[x2_name]
    x1 = np.linspace(x1[0], x1[-1], int(x1.size * resolution_factor), dtype="float64")
    x2 = np.linspace(x2[0], x2[-1], int(x2.size * resolution_factor), dtype="float64")
    X_t_normalised = X_t_normalised.interp(
        **{x1_name: x1, x2_name: x2}, method="nearest"
    )
    return X_t_normalised


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

    def stddev(self, task: Task):
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
    """

    def __init__(
        self,
        data_processor: Optional[DataProcessor] = None,
        task_loader: Optional[TaskLoader] = None,
    ):
        """
        Initialise DeepSensorModel.

        Args:
            data_processor (:class:`~.data.processor.DataProcessor`):
                DataProcessor object, used to unnormalise predictions.
            task_loader (:class:`~.data.loader.TaskLoader`):
                TaskLoader object, used to determine target variables for
                unnormalising.
        """
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

        ..
            TODO:
            - Test with multiple targets model

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

        if not unnormalise:
            X_t = X_t_normalised

        # Dict to store predictions for each target variable
        # Make this a subclass of dict like Task? And way to initialise cleanly with target_var_IDs?
        pred = Prediction(
            target_var_IDs, dates, X_t, X_t_mask, coord_names, n_samples=n_samples
        )

        def unnormalise_pred_array(arr, **kwargs):
            """Unnormalise an (N_dims, N_targets) array of predictions."""
            var_IDs_flattened = [
                var_ID
                for var_IDs in self.task_loader.target_var_IDs
                for var_ID in var_IDs
            ]
            assert arr.shape[0] == len(var_IDs_flattened)
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

            # If `DeepSensor` model child has been sub-classed with a `__call__` method,
            # we assume this is a distribution-like object that can be used to compute
            # mean, std and samples. Otherwise, run the model with `Task` for each prediction type.
            if hasattr(self, "__call__"):
                # Run model forwards once to generate output distribution, which we re-use
                dist = self(task, n_samples=n_samples)
                mean_arr = self.mean(dist)
                std_arr = self.stddev(dist)
                if n_samples >= 1:
                    B.set_random_seed(seed)
                    np.random.seed(seed)
                    if ar_sample:
                        samples_arr = self.ar_sample(
                            task,
                            n_samples=n_samples,
                            ar_subsample_factor=ar_subsample_factor,
                        )
                        samples_arr = samples_arr.reshape((n_samples, *mean_arr.shape))
                    else:
                        samples_arr = self.sample(dist, n_samples=n_samples)
            # Repeated code not ideal here...
            else:
                # Re-run model for each prediction type
                mean_arr = self.mean(task)
                std_arr = self.stddev(task)
                if n_samples >= 1:
                    B.set_random_seed(seed)
                    np.random.seed(seed)
                    if ar_sample:
                        samples_arr = self.ar_sample(
                            task,
                            n_samples=n_samples,
                            ar_subsample_factor=ar_subsample_factor,
                        )
                        samples_arr = samples_arr.reshape((n_samples, *mean_arr.shape))
                    else:
                        samples_arr = self.sample(task, n_samples=n_samples)

            # Concatenate multi-target predictions
            if isinstance(mean_arr, (list, tuple)):
                mean_arr = np.concatenate(mean_arr, axis=0)
                std_arr = np.concatenate(std_arr, axis=0)
                if n_samples >= 1:
                    # Axis 0 is sample dim, axis 1 is variable dim
                    samples_arr = np.concatenate(samples_arr, axis=1)

            # Unnormalise predictions
            if unnormalise:
                mean_arr = unnormalise_pred_array(mean_arr)
                std_arr = unnormalise_pred_array(std_arr, add_offset=False)
                if n_samples >= 1:
                    for sample_i in range(n_samples):
                        samples_arr[sample_i] = unnormalise_pred_array(
                            samples_arr[sample_i]
                        )

            pred.assign("mean", task["time"], mean_arr)
            pred.assign("std", task["time"], std_arr)
            if n_samples >= 1:
                pred.assign("samples", task["time"], samples_arr)

        if verbose:
            dur = time.time() - tic
            print(f"Done in {np.floor(dur / 60)}m:{dur % 60:.0f}s.\n")

        return pred


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
