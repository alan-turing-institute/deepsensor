import copy
from typing import List

import lab as B
import numpy as np
import xarray as xr

import deepsensor
from deepsensor.data.task import Task
from deepsensor.model.models import ConvNP


def concat_tasks(tasks: List[Task], multiple: int = 1) -> Task:
    """Concatenate a list of tasks into a single task containing multiple batches.

    Parameters
    ----------
    tasks : list of Task. List of tasks to concatenate into a single task.
    multiple : int. Contexts are padded to the smallest multiple of this number that is greater
        than the number of contexts in each task. Defaults to 1 (padded to the largest number of
        contexts in the tasks). Setting to a larger number will increase the amount of padding
        but decrease the range of tensor shapes presented to the model, which simplifies
        the computational graph in graph mode.

    Returns
    -------
    merged_task : Task. Task containing multiple batches.
    """
    if len(tasks) == 1:
        return tasks[0]

    contexts = []
    for i, task in enumerate(tasks):
        # Ensure converted to tensors with batch dims
        task = ConvNP.modify_task(task)
        tasks[i] = task

        # List of tuples of (x_c, y_c)
        contexts.append(list(zip(task["X_c"], task["Y_c"])))

    # List of tuples of merged (x_c, y_c) along batch dim with padding (w/ multiple=1000)
    merged_context = [
        deepsensor.backend.nps.merge_contexts(*[ci for ci in c], multiple=multiple)
        for c in zip(*contexts)
    ]

    merged_task = copy.deepcopy(tasks[0])

    # Convert list of tuples of (x_c, y_c) to list of x_c and list of y_c
    merged_task["X_c"] = [c[0] for c in merged_context]
    merged_task["Y_c"] = [c[1] for c in merged_context]

    merged_task["X_t"] = [
        B.concat(*[t["X_t"][i] for t in tasks], axis=0)
        for i in range(len(tasks[0]["X_t"]))
    ]
    merged_task["Y_t"] = [
        B.concat(*[t["Y_t"][i] for t in tasks], axis=0)
        for i in range(len(tasks[0]["Y_t"]))
    ]

    merged_task["time"] = [t["time"] for t in tasks]

    merged_task = Task(merged_task)

    return merged_task


def construct_x1x2_ds(gridded_ds):
    """
    Construct an xr.Dataset containing two vars, where each var is a 2D gridded channel whose
    values contain the x_1 and x_2 coordinate values, respectively.
    """
    X1, X2 = np.meshgrid(gridded_ds.x1, gridded_ds.x2, indexing="ij")
    ds = xr.Dataset(
        coords={"x1": gridded_ds.x1, "x2": gridded_ds.x2},
        data_vars={"x1_arr": (("x1", "x2"), X1), "x2_arr": (("x1", "x2"), X2)},
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
