import copy
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

Timestamp = Union[str, pd.Timestamp, np.datetime64]


class Prediction(dict):
    """Object to store model predictions in a dictionary-like format.

    Maps from target variable IDs to xarray/pandas objects containing
    prediction parameters (depending on the output distribution of the model).

    For example, if the model outputs a Gaussian distribution, then the xarray/pandas
    objects in the ``Prediction`` will contain a ``mean`` and ``std``.

    If using a ``Prediction`` to store model samples, there is only a ``samples`` entry, and the
    xarray/pandas objects will have an additional ``sample`` dimension.

    Args:
        target_var_IDs (List[str])
            List of target variable IDs.
        dates (List[Union[str, pd.Timestamp]])
            List of dates corresponding to the predictions.
        X_t (:class:`xarray.Dataset` | :class:`xarray.DataArray` | :class:`pandas.DataFrame` | :class:`pandas.Series` | :class:`pandas.Index` | :class:`numpy:numpy.ndarray`)
            Target locations to predict at. Can be an xarray object containing
            on-grid locations or a pandas object containing off-grid locations.
        X_t_mask (:class:`xarray.Dataset` | :class:`xarray.DataArray`, optional)
            2D mask to apply to gridded ``X_t`` (zero/False will be NaNs). Will be interpolated
            to the same grid as ``X_t``. Default None (no mask).
        n_samples (int)
            Number of joint samples to draw from the model. If 0, will not
            draw samples. Default 0.
        forecasting_mode (bool)
            If True, stored forecast predictions with an init_time and lead_time dimension,
            and a valid_time coordinate. If False, stores prediction at t=0 only
            (i.e. spatial interpolation), with only a single time dimension. Default False.
        lead_times (List[pd.Timedelta], optional)
            List of lead times to store in predictions. Must be provided if
            forecasting_mode is True. Default None.
    """

    def __init__(
        self,
        target_var_IDs: List[str],
        pred_params: List[str],
        dates: List[Timestamp],
        X_t: Union[
            xr.Dataset,
            xr.DataArray,
            pd.DataFrame,
            pd.Series,
            pd.Index,
            np.ndarray,
        ],
        X_t_mask: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        coord_names: dict = None,
        n_samples: int = 0,
        forecasting_mode: bool = False,
        lead_times: Optional[List[pd.Timedelta]] = None,
    ):
        self.target_var_IDs = target_var_IDs
        self.X_t_mask = X_t_mask
        if coord_names is None:
            coord_names = {"x1": "x1", "x2": "x2"}
        self.x1_name = coord_names["x1"]
        self.x2_name = coord_names["x2"]

        self.forecasting_mode = forecasting_mode
        if forecasting_mode:
            assert (
                lead_times is not None
            ), "If forecasting_mode is True, lead_times must be provided."
        self.lead_times = lead_times

        self.mode = infer_prediction_modality_from_X_t(X_t)

        self.pred_params = pred_params
        if n_samples >= 1:
            self.pred_params = [
                *pred_params,
                *[f"sample_{i}" for i in range(n_samples)],
            ]

        # Create empty xarray/pandas objects to store predictions
        if self.mode == "on-grid":
            for var_ID in self.target_var_IDs:
                if self.forecasting_mode:
                    prepend_dims = ["lead_time"]
                    prepend_coords = {"lead_time": lead_times}
                else:
                    prepend_dims = None
                    prepend_coords = None
                self[var_ID] = create_empty_spatiotemporal_xarray(
                    X_t,
                    dates,
                    data_vars=self.pred_params,
                    coord_names=coord_names,
                    prepend_dims=prepend_dims,
                    prepend_coords=prepend_coords,
                )
                if self.forecasting_mode:
                    self[var_ID] = self[var_ID].rename(time="init_time")
            if self.X_t_mask is None:
                # Create 2D boolean array of True values to simplify indexing
                self.X_t_mask = (
                    create_empty_spatiotemporal_xarray(X_t, dates[0:1], coord_names)
                    .to_array()
                    .isel(time=0, variable=0)
                    .astype(bool)
                )
        elif self.mode == "off-grid":
            # Repeat target locs for each date to create multiindex
            if self.forecasting_mode:
                index_names = ["lead_time", "init_time", *X_t.index.names]
                idxs = [
                    (lt, date, *idxs)
                    for lt in lead_times
                    for date in dates
                    for idxs in X_t.index
                ]
            else:
                index_names = ["time", *X_t.index.names]
                idxs = [(date, *idxs) for date in dates for idxs in X_t.index]
            index = pd.MultiIndex.from_tuples(idxs, names=index_names)
            for var_ID in self.target_var_IDs:
                self[var_ID] = pd.DataFrame(index=index, columns=self.pred_params)

    def __getitem__(self, key):
        # Support self[i] syntax
        if isinstance(key, int):
            key = self.target_var_IDs[key]
        return super().__getitem__(key)

    def __str__(self):
        dict_repr = {var_ID: self.pred_params for var_ID in self.target_var_IDs}
        return f"Prediction({dict_repr}), mode={self.mode}"

    def assign(
        self,
        prediction_parameter: str,
        date: Union[str, pd.Timestamp],
        data: np.ndarray,
        lead_times: Optional[List[pd.Timedelta]] = None,
    ):
        """Args:
        prediction_parameter (str)
            ...
        date (Union[str, pd.Timestamp])
            ...
        data (np.ndarray)
            If off-grid: Shape (N_var, N_targets) or (N_samples, N_var, N_targets).
            If on-grid: Shape (N_var, N_x1, N_x2) or (N_samples, N_var, N_x1, N_x2).
        lead_time (pd.Timedelta, optional)
            Lead time of the forecast. Required if forecasting_mode is True. Default None.
        """
        if self.forecasting_mode:
            assert (
                lead_times is not None
            ), "If forecasting_mode is True, lead_times must be provided."

            msg = f"""
            If forecasting_mode is True, lead_times must be of equal length to the number of
            variables in the data (the first dimension). Got {lead_times=} of length
            {len(lead_times)} lead times and data shape {data.shape}.
            """
            assert len(lead_times) == data.shape[0], msg

        if self.mode == "on-grid":
            if prediction_parameter != "samples":
                for i, (var_ID, pred) in enumerate(zip(self.target_var_IDs, data)):
                    if self.forecasting_mode:
                        index = (lead_times[i], date)
                    else:
                        index = date
                    self[var_ID][prediction_parameter].loc[index].data[
                        self.X_t_mask.data
                    ] = pred.ravel()
            elif prediction_parameter == "samples":
                assert len(data.shape) == 4, (
                    f"If prediction_parameter is 'samples', and mode is 'on-grid', data must"
                    f"have shape (N_samples, N_var, N_x1, N_x2). Got {data.shape}."
                )
                for sample_i, sample in enumerate(data):
                    for i, (var_ID, pred) in enumerate(
                        zip(self.target_var_IDs, sample)
                    ):
                        if self.forecasting_mode:
                            index = (lead_times[i], date)
                        else:
                            index = date
                        self[var_ID][f"sample_{sample_i}"].loc[index].data[
                            self.X_t_mask.data
                        ] = pred.ravel()

        elif self.mode == "off-grid":
            if prediction_parameter != "samples":
                for i, (var_ID, pred) in enumerate(zip(self.target_var_IDs, data)):
                    if self.forecasting_mode:
                        index = (lead_times[i], date)
                    else:
                        index = date
                    self[var_ID].loc[index, prediction_parameter] = pred
            elif prediction_parameter == "samples":
                assert len(data.shape) == 3, (
                    f"If prediction_parameter is 'samples', and mode is 'off-grid', data must"
                    f"have shape (N_samples, N_var, N_targets). Got {data.shape}."
                )
                for sample_i, sample in enumerate(data):
                    for i, (var_ID, pred) in enumerate(
                        zip(self.target_var_IDs, sample)
                    ):
                        if self.forecasting_mode:
                            index = (lead_times[i], date)
                        else:
                            index = date
                        self[var_ID].loc[index, f"sample_{sample_i}"] = pred


def create_empty_spatiotemporal_xarray(
    X: Union[xr.Dataset, xr.DataArray],
    dates: List[Timestamp],
    coord_names: dict = None,
    data_vars: List[str] = None,
    prepend_dims: Optional[List[str]] = None,
    prepend_coords: Optional[dict] = None,
):
    """...

    Args:
        X (:class:`xarray.Dataset` | :class:`xarray.DataArray`):
            ...
        dates (List[...]):
            ...
        coord_names (dict, optional):
            Dict mapping from normalised coord names to raw coord names,
            by default {"x1": "x1", "x2": "x2"}
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
    if coord_names is None:
        coord_names = {"x1": "x1", "x2": "x2"}
    if data_vars is None:
        data_vars = ["var"]

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

    return pred_ds


def increase_spatial_resolution(
    X_t_normalised,
    resolution_factor,
    coord_names: dict = None,
):
    """...

    ..
        # TODO wasteful to interpolate X_t_normalised

    Args:
        X_t_normalised (...):
            ...
        resolution_factor (...):
            ...
        coord_names (dict, optional):
            Dict mapping from normalised coord names to raw coord names,
            by default {"x1": "x1", "x2": "x2"}

    Returns:
        ...
            ...

    """
    assert isinstance(resolution_factor, (float, int))
    assert isinstance(X_t_normalised, (xr.DataArray, xr.Dataset))
    if coord_names is None:
        coord_names = {"x1": "x1", "x2": "x2"}
    x1_name, x2_name = coord_names["x1"], coord_names["x2"]
    x1, x2 = X_t_normalised.coords[x1_name], X_t_normalised.coords[x2_name]
    x1 = np.linspace(x1[0], x1[-1], int(x1.size * resolution_factor), dtype="float64")
    x2 = np.linspace(x2[0], x2[-1], int(x2.size * resolution_factor), dtype="float64")
    X_t_normalised = X_t_normalised.interp(
        **{x1_name: x1, x2_name: x2}, method="nearest"
    )
    return X_t_normalised


def infer_prediction_modality_from_X_t(
    X_t: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series, pd.Index, np.ndarray],
) -> str:
    """Args:
        X_t (Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series, pd.Index, np.ndarray]):
            ...

    Returns:
        str: "on-grid" if X_t is an xarray object, "off-grid" if X_t is a pandas or numpy object.

    Raises:
        ValueError
            If X_t is not an xarray, pandas or numpy object.
    """
    if isinstance(X_t, (xr.DataArray, xr.Dataset)):
        mode = "on-grid"
    elif isinstance(X_t, (pd.DataFrame, pd.Series, pd.Index, np.ndarray)):
        mode = "off-grid"
    else:
        raise ValueError(
            f"X_t must be and xarray, pandas or numpy object. Got {type(X_t)}."
        )
    return mode


def _get_coordinate_extent(
    ds: Union[xr.DataArray, xr.Dataset],
    orig_x1_name: str,
    orig_x2_name: str,
    x1_ascend: bool,
    x2_ascend: bool,
) -> Tuple:
    """Get coordinate extent of dataset. This method is applied to either X_t or patchwise predictions.

    Parameters
    ----------
    ds : Data object
        The dataset or data array to determine coordinate extent for.

    x1_ascend : bool
        Whether the x1 coordinates ascend (increase) from top to bottom.

    x2_ascend : bool
        Whether the x2 coordinates ascend (increase) from left to right.

    Returns:
    -------
    tuple of tuples:
        Extents of x1 and x2 coordinates as ((min_x1, max_x1), (min_x2, max_x2)).
    """
    if x1_ascend:
        ds_x1_coords = (
            ds.coords[orig_x1_name].min().values,
            ds.coords[orig_x1_name].max().values,
        )
    else:
        ds_x1_coords = (
            ds.coords[orig_x1_name].max().values,
            ds.coords[orig_x1_name].min().values,
        )
    if x2_ascend:
        ds_x2_coords = (
            ds.coords[orig_x2_name].min().values,
            ds.coords[orig_x2_name].max().values,
        )
    else:
        ds_x2_coords = (
            ds.coords[orig_x2_name].max().values,
            ds.coords[orig_x2_name].min().values,
        )
    return ds_x1_coords, ds_x2_coords


def _get_index(
    *args,
    X_t: Union[
        xr.Dataset,
        xr.DataArray,
        pd.DataFrame,
        pd.Series,
        pd.Index,
        np.ndarray,
    ],
    orig_x1_name: str,
    orig_x2_name: str,
    x1: bool = True,
) -> Union[int, Tuple[List[int], List[int]]]:
    """Convert coordinates into pixel row/column (index).

    Parameters
    ----------
    args : tuple
        If one argument (numeric), it represents the coordinate value.
        If two arguments (lists), they represent lists of coordinate values.

    X_t : (:class:`xarray.Dataset` | :class:`xarray.DataArray` | :class:`pandas.DataFrame` | :class:`pandas.Series` | :class:`pandas.Index` | :class:`numpy:numpy.ndarray`)
        Target locations to predict at. Can be an xarray object
        containing on-grid locations or a pandas object containing off-grid locations.

    x1 : bool, optional
        If True, compute index for x1 (default is True).

    Returns:
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
                np.abs(X_t.coords[orig_x1_name].values - patch_coord)
            )
        else:
            coord_index = np.argmin(
                np.abs(X_t.coords[orig_x2_name].values - patch_coord)
            )
        return coord_index

    elif len(args) == 2:
        patch_x1, patch_x2 = args
        x1_index = [
            np.argmin(np.abs(X_t.coords[orig_x1_name].values - target_x1))
            for target_x1 in patch_x1
        ]
        x2_index = [
            np.argmin(np.abs(X_t.coords[orig_x2_name].values - target_x2))
            for target_x2 in patch_x2
        ]
        return (x1_index, x2_index)


def stitch_clipped_predictions(
    patch_preds: List[Prediction],
    patch_overlap: int,
    patches_per_row: int,
    X_t: Union[
        xr.Dataset,
        xr.DataArray,
        pd.DataFrame,
        pd.Series,
        pd.Index,
        np.ndarray,
    ],
    orig_x1_name: str,
    orig_x2_name: str,
    x1_ascend: bool = True,
    x2_ascend: bool = True,
) -> Prediction:
    """Stitch patchwise predictions to form prediction at original extent.

    Parameters
    ----------
    patch_preds : list (class:`~.model.pred.Prediction`)
        List of patchwise predictions

    patch_overlap: int
        Overlap between adjacent patches in pixels.

    patches_per_row: int
        Number of patchwise predictions in each row.

    X_t : (:class:`xarray.Dataset` | :class:`xarray.DataArray` | :class:`pandas.DataFrame` | :class:`pandas.Series` | :class:`pandas.Index` | :class:`numpy:numpy.ndarray`)
        Target locations to predict at. Can be an xarray object
        containing on-grid locations or a pandas object containing off-grid locations.

    orig_x1_name : str
        x1 coordinate names of original unnormalised dataset

    orig_x2_name : str
        x2 coordinate names of original unnormalised dataset

    x1_ascend : bool
        Boolean defining whether the x1 coords ascend (increase) from top to bottom, default = True.

    x2_ascend : bool
        Boolean defining whether the x2 coords ascend (increase) from left to right, default = True.

    Returns:
    -------
    combined: dict
        Dictionary object containing the stitched model predictions.
    """
    # Get row/col index values of X_t.
    data_x1_coords, data_x2_coords = _get_coordinate_extent(
        X_t,
        orig_x1_name=orig_x1_name,
        orig_x2_name=orig_x2_name,
        x1_ascend=x1_ascend,
        x2_ascend=x2_ascend,
    )
    data_x1_index, data_x2_index = _get_index(
        data_x1_coords,
        data_x2_coords,
        X_t=X_t,
        orig_x1_name=orig_x1_name,
        orig_x2_name=orig_x2_name,
    )

    # Iterate through patchwise predictions and slice edges prior to stitchin.
    patches_clipped = []
    for i, patch_pred in enumerate(patch_preds):
        # get one variable name to use for coordinates and extent
        first_key = list(patch_pred.keys())[0]
        # Get row/col index values of each patch.
        patch_x1_coords, patch_x2_coords = _get_coordinate_extent(
            patch_pred[first_key],
            orig_x1_name=orig_x1_name,
            orig_x2_name=orig_x2_name,
            x1_ascend=x1_ascend,
            x2_ascend=x2_ascend,
        )
        patch_x1_index, patch_x2_index = _get_index(
            patch_x1_coords,
            patch_x2_coords,
            X_t=X_t,
            orig_x1_name=orig_x1_name,
            orig_x2_name=orig_x2_name,
        )

        # Calculate size of border to slice of each edge of patchwise predictions.
        # Initially set the size of all borders to the size of the overlap.
        b_x1_min, b_x1_max = patch_overlap[0], patch_overlap[0]
        b_x2_min, b_x2_max = patch_overlap[1], patch_overlap[1]

        # Do not remove border for the patches along top and left of dataset and change overlap size for last patch in each row and column.
        if patch_x2_index[0] == data_x2_index[0]:
            b_x2_min = 0
            b_x2_max = b_x2_max

        # At end of row (when patch_x2_index = data_x2_index), calculate the number of pixels to remove from left hand side of patch.
        elif patch_x2_index[1] == data_x2_index[1]:
            b_x2_max = 0
            patch_row_prev = patch_preds[i - 1]

            # If x2 is ascending, subtract previous patch x2 max value from current patch x2 min value to get bespoke overlap in column pixels.
            # To account for the clipping done to the previous patch, then subtract patch_overlap value in pixels
            if x2_ascend:
                prev_patch_x2_max = _get_index(
                    patch_row_prev[first_key].coords[orig_x2_name].max(),
                    X_t=X_t,
                    orig_x1_name=orig_x1_name,
                    orig_x2_name=orig_x2_name,
                    x1=False,
                )
                b_x2_min = (prev_patch_x2_max - patch_x2_index[0]) - patch_overlap[1]

            # If x2 is descending, subtract current patch max x2 value from previous patch min x2 value to get bespoke overlap in column pixels.
            # To account for the clipping done to the previous patch, then subtract patch_overlap value in pixels
            else:
                prev_patch_x2_min = _get_index(
                    patch_row_prev[first_key].coords[orig_x2_name].min(),
                    X_t=X_t,
                    orig_x1_name=orig_x1_name,
                    orig_x2_name=orig_x2_name,
                    x1=False,
                )
                b_x2_min = (patch_x2_index[0] - prev_patch_x2_min) - patch_overlap[1]
        else:
            b_x2_max = b_x2_max

        # Repeat process as above for x1 coordinates.
        if patch_x1_index[0] == data_x1_index[0]:
            b_x1_min = 0

        elif abs(patch_x1_index[1] - data_x1_index[1]) < 2:
            b_x1_max = 0
            b_x1_max = b_x1_max
            patch_prev = patch_preds[i - patches_per_row]
            if x1_ascend:
                prev_patch_x1_max = _get_index(
                    patch_prev[first_key].coords[orig_x1_name].max(),
                    X_t=X_t,
                    orig_x1_name=orig_x1_name,
                    orig_x2_name=orig_x2_name,
                    x1=True,
                )
                b_x1_min = (prev_patch_x1_max - patch_x1_index[0]) - patch_overlap[0]
            else:
                prev_patch_x1_min = _get_index(
                    patch_prev[first_key].coords[orig_x1_name].min(),
                    X_t=X_t,
                    orig_x1_name=orig_x1_name,
                    orig_x2_name=orig_x2_name,
                    x1=True,
                )

                b_x1_min = (prev_patch_x1_min - patch_x1_index[0]) - patch_overlap[0]
        else:
            b_x1_max = b_x1_max

        patch_clip_x1_min = int(b_x1_min)
        patch_clip_x1_max = int(patch_pred[first_key].sizes[orig_x1_name] - b_x1_max)
        patch_clip_x2_min = int(b_x2_min)
        patch_clip_x2_max = int(patch_pred[first_key].sizes[orig_x2_name] - b_x2_max)

        # Define slicing parameters
        slicing_params = {
            orig_x1_name: slice(patch_clip_x1_min, patch_clip_x1_max),
            orig_x2_name: slice(patch_clip_x2_min, patch_clip_x2_max),
        }

        # Slice patchwise predictions
        patch_clip = {
            key: dataset.isel(**slicing_params) for key, dataset in patch_pred.items()
        }

        patches_clipped.append(patch_clip)

    # Create blank prediction object to stitch prediction values onto.
    stitched_prediction = copy.deepcopy(patch_preds[0])
    # Set prediction object extent to the same as X_t.
    for var_name, data_array in stitched_prediction.items():
        blank_ds = xr.Dataset(
            coords={
                orig_x1_name: X_t[orig_x1_name],
                orig_x2_name: X_t[orig_x2_name],
                "time": stitched_prediction[0]["time"],
            }
        )

        # Set data variable names e.g. mean, std to those in patched prediction. Make all values Nan.
        for data_var in data_array.data_vars:
            blank_ds[data_var] = data_array[data_var]
            blank_ds[data_var][:] = np.nan
        stitched_prediction[var_name] = blank_ds

    # Restructure prediction objects for merging
    restructured_patches = {
        key: [item[key] for item in patches_clipped]
        for key in patches_clipped[0].keys()
    }

    # Merge patchwise predictions to create final stiched prediction.
    # Iterate over each variable (key) in the prediction dictionary
    for var_name, patches in restructured_patches.items():
        # Retrieve the blank dataset for the current variable
        prediction_array = stitched_prediction[var_name]

        # Merge each patch into the combined dataset
        for patch in patches:
            for var in patch.data_vars:
                # Reindex the patch to catch any slight rounding errors and misalignment with the combined dataset
                reindexed_patch = patch[var].reindex_like(
                    prediction_array[var], method="nearest", tolerance=1e-6
                )

                # Combine data, prioritizing non-NaN values from patches
                prediction_array[var] = prediction_array[var].where(
                    np.isnan(reindexed_patch), reindexed_patch
                )

        # Update the dictionary with the merged dataset
        stitched_prediction[var_name] = prediction_array
    return stitched_prediction
