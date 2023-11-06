from typing import Union, List, Optional

import numpy as np
import pandas as pd
import xarray as xr


class Prediction(dict):
    """
    Object to store model predictions in a dictionary-like format.

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
    """

    def __init__(
        self,
        target_var_IDs: List[str],
        dates: List[Union[str, pd.Timestamp]],
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
    ):
        self.target_var_IDs = target_var_IDs
        self.X_t_mask = X_t_mask
        if coord_names is None:
            coord_names = {"x1": "x1", "x2": "x2"}
        self.x1_name = coord_names["x1"]
        self.x2_name = coord_names["x2"]

        self.mode = infer_prediction_modality_from_X_t(X_t)

        # TODO don't assume Gaussian distribution
        self.pred_parameters = ["mean", "std"]
        if n_samples >= 1:
            self.pred_parameters.extend([f"sample_{i}" for i in range(n_samples)])

        if self.mode == "on-grid":
            for var_ID in self.target_var_IDs:
                # Create empty xarray/pandas objects to store predictions
                self[var_ID] = create_empty_spatiotemporal_xarray(
                    X_t,
                    dates,
                    data_vars=self.pred_parameters,
                    coord_names=coord_names,
                )
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
            idxs = [(date, *idxs) for date in dates for idxs in X_t.index]
            index = pd.MultiIndex.from_tuples(idxs, names=["time", *X_t.index.names])
            for var_ID in self.target_var_IDs:
                self[var_ID] = pd.DataFrame(index=index, columns=self.pred_parameters)

    def __getitem__(self, key):
        # Support self[i] syntax
        if isinstance(key, int):
            key = self.target_var_IDs[key]
        return super().__getitem__(key)

    def __str__(self):
        dict_repr = {var_ID: self.pred_parameters for var_ID in self.target_var_IDs}
        return f"Prediction({dict_repr}), mode={self.mode}"

    def assign(
        self,
        prediction_parameter: str,
        date: Union[str, pd.Timestamp],
        data: np.ndarray,
    ):
        """

        Args:
            prediction_parameter (str)
                ...
            date (Union[str, pd.Timestamp])
                ...
            data (np.ndarray)
                If off-grid: Shape (N_var, N_targets) or (N_samples, N_var, N_targets).
                If on-grid: Shape (N_var, N_x1, N_x2) or (N_samples, N_var, N_x1, N_x2).
        """
        if self.mode == "on-grid":
            if prediction_parameter != "samples":
                for var_ID, pred in zip(self.target_var_IDs, data):
                    self[var_ID][prediction_parameter].loc[date].data[
                        self.X_t_mask.data
                    ] = pred.ravel()
            elif prediction_parameter == "samples":
                assert len(data.shape) == 4, (
                    f"If prediction_parameter is 'samples', and mode is 'on-grid', data must"
                    f"have shape (N_samples, N_var, N_x1, N_x2). Got {data.shape}."
                )
                for sample_i, sample in enumerate(data):
                    for var_ID, pred in zip(self.target_var_IDs, sample):
                        self[var_ID][f"sample_{sample_i}"].loc[date].data[
                            self.X_t_mask.data
                        ] = pred.ravel()

        elif self.mode == "off-grid":
            if prediction_parameter != "samples":
                for var_ID, pred in zip(self.target_var_IDs, data):
                    self[var_ID][prediction_parameter].loc[date] = pred
            elif prediction_parameter == "samples":
                assert len(data.shape) == 3, (
                    f"If prediction_parameter is 'samples', and mode is 'off-grid', data must"
                    f"have shape (N_samples, N_var, N_targets). Got {data.shape}."
                )
                for sample_i, sample in enumerate(data):
                    for var_ID, pred in zip(self.target_var_IDs, sample):
                        self[var_ID][f"sample_{sample_i}"].loc[date] = pred


def create_empty_spatiotemporal_xarray(
    X: Union[xr.Dataset, xr.DataArray],
    dates: List,
    coord_names: dict = None,
    data_vars: List[str] = None,
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

    # TODO: Convert init time to forecast time?
    # pred_ds = pred_ds.assign_coords(
    #     time=pred_ds['time'] + pd.Timedelta(days=task_loader.target_delta_t[0]))

    return pred_ds


def increase_spatial_resolution(
    X_t_normalised,
    resolution_factor,
    coord_names: dict = None,
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
    X_t: Union[xr.DataArray, xr.Dataset, pd.DataFrame, pd.Series, pd.Index, np.ndarray]
) -> str:
    """

    Args:
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
