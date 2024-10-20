import xarray as xr
from deepsensor.model.pred import Prediction


def compute_errors(pred: Prediction, target: xr.Dataset) -> xr.Dataset:
    """
    Compute errors between predictions and targets.

    Args:
        pred: Prediction object.
        target: Target data.

    Returns:
        xr.Dataset: Dataset of pointwise differences between predictions and targets
        at the same valid time in the predictions. Note, the difference is positive
        when the prediction is greater than the target.
    """
    errors = {}
    for var_ID, pred_var in pred.items():
        target_var = target[var_ID]
        error = pred_var["mean"] - target_var.sel(time=pred_var.time)
        error.name = f"{var_ID}"
        errors[var_ID] = error
    return xr.Dataset(errors)
