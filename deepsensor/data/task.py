import deepsensor

from typing import Callable, Union, Tuple, List, Optional
import numpy as np
import lab as B
import plum
import copy

from ..errors import TaskSetIndexError, GriddedDataError


class Task(dict):
    """Task dictionary class.

    Inherits from ``dict`` and adds methods for printing and modifying the
    data.

    Args:
        task_dict (dict):
            Dictionary containing the task.
    """

    def __init__(self, task_dict: dict) -> None:
        super().__init__(task_dict)

        if "ops" not in self:
            # List of operations (str) to indicate how the task has been modified
            #   (reshaping, casting, etc)
            self["ops"] = []

    @classmethod
    def summarise_str(cls, k, v):
        """Return string summaries for the _str__ method."""
        if plum.isinstance(v, B.Numeric):
            return v.shape
        elif plum.isinstance(v, tuple):
            return tuple(vi.shape for vi in v)
        elif plum.isinstance(v, list):
            return [cls.summarise_str(k, vi) for vi in v]
        else:
            return v

    @classmethod
    def summarise_repr(cls, k, v) -> str:
        """Summarise the task in a representation that can be printed.

        Args:
            cls (:class:`deepsensor.data.task.Task`:):
                Task class.
            k (str):
                Key of the task dictionary.
            v (object):
                Value of the task dictionary.

        Returns:
            str: String representation of the task.
        """
        if v is None:
            return "None"
        elif plum.isinstance(v, B.Numeric):
            return f"{type(v).__name__}/{v.dtype}/{v.shape}"
        if plum.isinstance(v, deepsensor.backend.nps.mask.Masked):
            return f"{type(v).__name__}/(y={v.y.dtype}/{v.y.shape})/(mask={v.mask.dtype}/{v.mask.shape})"
        elif plum.isinstance(v, tuple):
            # return tuple(vi.shape for vi in v)
            return tuple([cls.summarise_repr(k, vi) for vi in v])
        elif plum.isinstance(v, list):
            return [cls.summarise_repr(k, vi) for vi in v]
        else:
            return f"{type(v).__name__}/{v}"

    def __str__(self) -> str:
        """Print a convenient summary of the task dictionary.

        For array entries, print their shape, otherwise print the value.
        """
        s = ""
        for k, v in self.items():
            if v is None:
                continue
            s += f"{k}: {Task.summarise_str(k, v)}\n"
        return s

    def __repr__(self) -> str:
        """Print a convenient summary of the task dictionary.

        Print the type of each entry and if it is an array, print its shape,
        otherwise print the value.
        """
        s = ""
        for k, v in self.items():
            s += f"{k}: {Task.summarise_repr(k, v)}\n"
        return s

    def op(self, f: Callable, op_flag: Optional[str] = None):
        """Apply function f to the array elements of a task dictionary.

        Useful for recasting to a different dtype or reshaping (e.g. adding a
        batch dimension).

        Args:
            f (callable):
                Function to apply to the array elements of the task.
            op_flag (str):
                Flag to set in the task dictionary's `ops` key.

        Returns:
            :class:`deepsensor.data.task.Task`:
                Task with f applied to the array elements and op_flag set in
                the ``ops`` key.
        """

        def recurse(k, v):
            if type(v) is list:
                return [recurse(k, vi) for vi in v]
            elif type(v) is tuple:
                return (recurse(k, v[0]), recurse(k, v[1]))
            elif isinstance(
                v,
                (np.ndarray, np.ma.MaskedArray, deepsensor.backend.nps.Masked),
            ):
                return f(v)
            else:
                return v  # covers metadata entries

        self = copy.deepcopy(self)  # don't modify the original
        for k, v in self.items():
            self[k] = recurse(k, v)
        self["ops"].append(op_flag)

        return self  # altered by reference, but return anyway

    def add_batch_dim(self):
        """Add a batch dimension to the arrays in the task dictionary.

        Returns:
            :class:`deepsensor.data.task.Task`:
                Task with batch dimension added to the array elements.
        """
        return self.op(lambda x: x[None, ...], op_flag="batch_dim")

    def cast_to_float32(self):
        """Cast the arrays in the task dictionary to float32.

        Returns:
            :class:`deepsensor.data.task.Task`:
                Task with arrays cast to float32.
        """
        return self.op(lambda x: x.astype(np.float32), op_flag="float32")

    def flatten_gridded_data(self):
        """Convert any gridded data in ``Task`` to flattened arrays.

        Necessary for AR sampling, which doesn't yet permit gridded context sets.

        Args:
            task : :class:`~.data.task.Task`
                ...

        Returns:
            :class:`deepsensor.data.task.Task`:
                ...
        """
        self["X_c"] = [flatten_X(X) for X in self["X_c"]]
        self["Y_c"] = [flatten_Y(Y) for Y in self["Y_c"]]
        if self["X_t"] is not None:
            self["X_t"] = [flatten_X(X) for X in self["X_t"]]
        if self["Y_t"] is not None:
            self["Y_t"] = [flatten_Y(Y) for Y in self["Y_t"]]

        self["ops"].append("gridded_data_flattened")

        return self

    def remove_context_nans(self):
        """If NaNs are present in task["Y_c"], remove them (and corresponding task["X_c"]).

        Returns:
            :class:`deepsensor.data.task.Task`:
                ...
        """
        if "batch_dim" in self["ops"]:
            raise ValueError(
                "Cannot remove NaNs from task if a batch dim has been added."
            )

        # First check whether there are any NaNs that we need to remove
        nans_present = False
        for Y_c in self["Y_c"]:
            if B.any(B.isnan(Y_c)):
                nans_present = True
                break

        if not nans_present:
            return self

        # NaNs present in self - remove NaNs
        for i, (X, Y) in enumerate(zip(self["X_c"], self["Y_c"])):
            Y_c_nans = B.isnan(Y)
            if B.any(Y_c_nans):
                if isinstance(X, tuple):
                    # Gridded data - need to flatten to remove NaNs
                    X = flatten_X(X)
                    Y = flatten_Y(Y)
                    Y_c_nans = flatten_Y(Y_c_nans)
                Y_c_nans = B.any(Y_c_nans, axis=0)  # shape (n_cargets,)
                self["X_c"][i] = X[:, ~Y_c_nans]
                self["Y_c"][i] = Y[:, ~Y_c_nans]

        self["ops"].append("context_nans_removed")

        return self

    def remove_target_nans(self):
        """If NaNs are present in task["Y_t"], remove them (and corresponding task["X_t"]).

        Returns:
            :class:`deepsensor.data.task.Task`:
                ...
        """
        if "batch_dim" in self["ops"]:
            raise ValueError(
                "Cannot remove NaNs from task if a batch dim has been added."
            )

        # First check whether there are any NaNs that we need to remove
        nans_present = False
        for Y_t in self["Y_t"]:
            if B.any(B.isnan(Y_t)):
                nans_present = True
                break

        if not nans_present:
            return self

        # NaNs present in self - remove NaNs
        for i, (X, Y) in enumerate(zip(self["X_t"], self["Y_t"])):
            Y_t_nans = B.isnan(Y)
            if "Y_t_aux" in self.keys():
                self["Y_t_aux"] = flatten_Y(self["Y_t_aux"])
            if B.any(Y_t_nans):
                if isinstance(X, tuple):
                    # Gridded data - need to flatten to remove NaNs
                    X = flatten_X(X)
                    Y = flatten_Y(Y)
                    Y_t_nans = flatten_Y(Y_t_nans)
                Y_t_nans = B.any(Y_t_nans, axis=0)  # shape (n_targets,)
                self["X_t"][i] = X[:, ~Y_t_nans]
                self["Y_t"][i] = Y[:, ~Y_t_nans]
                if "Y_t_aux" in self.keys():
                    self["Y_t_aux"] = self["Y_t_aux"][:, ~Y_t_nans]

        self["ops"].append("target_nans_removed")

        return self

    def mask_nans_numpy(self):
        """Replace NaNs with zeroes and set a mask to indicate where the NaNs
        were.

        Returns:
            :class:`deepsensor.data.task.Task`:
                Task with NaNs set to zeros and a mask indicating where the
                missing values are.
        """
        if "batch_dim" not in self["ops"]:
            raise ValueError("Must call `add_batch_dim` before `mask_nans_numpy`")

        def f(arr):
            if isinstance(arr, deepsensor.backend.nps.Masked):
                nps_mask = arr.mask == 0
                nan_mask = np.isnan(arr.y)
                mask = np.logical_or(nps_mask, nan_mask)
                mask = np.any(mask, axis=1, keepdims=True)
                data = arr.y
                data[nan_mask] = 0.0
                arr = deepsensor.backend.nps.Masked(data, mask)
            else:
                mask = np.isnan(arr)
                if np.any(mask):
                    # arr = np.ma.MaskedArray(arr, mask=mask, fill_value=0.0)
                    arr = np.ma.fix_invalid(arr, fill_value=0.0)
            return arr

        return self.op(lambda x: f(x), op_flag="numpy_mask")

    def mask_nans_nps(self):
        """...

        Returns:
            :class:`deepsensor.data.task.Task`:
                ...
        """
        if "batch_dim" not in self["ops"]:
            raise ValueError("Must call `add_batch_dim` before `mask_nans_nps`")
        if "numpy_mask" not in self["ops"]:
            raise ValueError("Must call `mask_nans_numpy` before `mask_nans_nps`")

        def f(arr):
            if isinstance(arr, np.ma.MaskedArray):
                # Mask array (True for observed, False for missing). Keep size 1 variable dim.
                mask = ~B.any(arr.mask, axis=1, squeeze=False)
                mask = B.cast(B.dtype(arr.data), mask)
                arr = deepsensor.backend.nps.Masked(arr.data, mask)
            return arr

        return self.op(lambda x: f(x), op_flag="nps_mask")

    def convert_to_tensor(self):
        """Convert to tensor object based on deep learning backend.

        Returns:
            :class:`deepsensor.data.task.Task`:
                Task with arrays converted to deep learning tensor objects.
        """

        def f(arr):
            if isinstance(arr, deepsensor.backend.nps.Masked):
                arr = deepsensor.backend.nps.Masked(
                    deepsensor.backend.convert_to_tensor(arr.y),
                    deepsensor.backend.convert_to_tensor(arr.mask),
                )
            else:
                arr = deepsensor.backend.convert_to_tensor(arr)
            return arr

        return self.op(lambda x: f(x), op_flag="tensor")


def append_obs_to_task(
    task: Task,
    X_new: B.Numeric,
    Y_new: B.Numeric,
    context_set_idx: int,
):
    """Append a single observation to a context set in ``task``.

    Makes a deep copy of the data structure to avoid affecting the original
    object.

    ..
        TODO: for speed during active learning algs, consider a shallow copy
        option plus ability to remove observations.

    Args:
        task (:class:`deepsensor.data.task.Task`:): The task to modify.
        X_new (array-like): New observation coordinates.
        Y_new (array-like): New observation values.
        context_set_idx (int): Index of the context set to append to.

    Returns:
        :class:`deepsensor.data.task.Task`:
            Task with new observation appended to the context set.
    """
    if not 0 <= context_set_idx <= len(task["X_c"]) - 1:
        raise TaskSetIndexError(context_set_idx, len(task["X_c"]), "context")

    if isinstance(task["X_c"][context_set_idx], tuple):
        raise GriddedDataError("Cannot append to gridded data")

    task_with_new = copy.deepcopy(task)

    if Y_new.ndim == 0:
        # Add size-1 observation and data dimension
        Y_new = Y_new[None, None]

    # Add size-1 observation dimension
    if X_new.ndim == 1:
        X_new = X_new[:, None]
    if Y_new.ndim == 1:
        Y_new = Y_new[:, None]

    # Context set with proposed latent sensors
    task_with_new["X_c"][context_set_idx] = np.concatenate(
        [task["X_c"][context_set_idx], X_new], axis=-1
    )

    # Append proxy observations
    task_with_new["Y_c"][context_set_idx] = np.concatenate(
        [task["Y_c"][context_set_idx], Y_new], axis=-1
    )

    return task_with_new


def flatten_X(X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Convert tuple of gridded coords to (2, N) array if necessary.

    Args:
        X (:class:`numpy:numpy.ndarray` | Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]):
            ...

    Returns:
        :class:`numpy:numpy.ndarray`
            ...
    """
    if type(X) is tuple:
        X1, X2 = np.meshgrid(X[0], X[1], indexing="ij")
        X = np.stack([X1.ravel(), X2.ravel()], axis=0)
    return X


def flatten_Y(Y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Convert gridded data of shape (N_dim, N_x1, N_x2) to (N_dim, N_x1 * N_x2)
    array if necessary.

    Args:
        Y (:class:`numpy:numpy.ndarray` | Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]):
            ...

    Returns:
        :class:`numpy:numpy.ndarray`
            ...
    """
    if Y.ndim == 3:
        Y = Y.reshape(*Y.shape[:-2], -1)
    return Y


def concat_tasks(tasks: List[Task], multiple: int = 1) -> Task:
    """Concatenate a list of tasks into a single task containing multiple batches.

    ..

    Todo:
        - Consider moving to ``nps.py`` as this leverages ``neuralprocesses``
          functionality.
        - Raise error if ``aux_t`` values passed (not supported I don't think)

    Args:
        tasks (List[:class:`deepsensor.data.task.Task`:]):
            List of tasks to concatenate into a single task.
        multiple (int, optional):
            Contexts are padded to the smallest multiple of this number that is
            greater than the number of contexts in each task. Defaults to 1
            (padded to the largest number of contexts in the tasks). Setting
            to a larger number will increase the amount of padding but decrease
            the range of tensor shapes presented to the model, which simplifies
            the computational graph in graph mode.

    Returns:
        :class:`~.data.task.Task`: Task containing multiple batches.

    Raises:
        ValueError:
            If the tasks have different numbers of target sets.
        ValueError:
            If the tasks have different numbers of targets.
        ValueError:
            If the tasks have different types of target sets (gridded/
            non-gridded).
    """
    if len(tasks) == 1:
        return tasks[0]

    for i, task in enumerate(tasks):
        if "numpy_mask" in task["ops"] or "nps_mask" in task["ops"]:
            raise ValueError(
                "Cannot concatenate tasks that have had NaNs masked. "
                "Masking will be applied automatically after concatenation."
            )
        if "target_nans_removed" not in task["ops"]:
            task = task.remove_target_nans()
        if "batch_dim" not in task["ops"]:
            task = task.add_batch_dim()
        if "float32" not in task["ops"]:
            task = task.cast_to_float32()
        tasks[i] = task

    # Assert number of target sets equal
    n_target_sets = [len(task["Y_t"]) for task in tasks]
    if not all([n == n_target_sets[0] for n in n_target_sets]):
        raise ValueError(
            f"All tasks must have the same number of target sets to concatenate: got {n_target_sets}. "
        )
    n_target_sets = n_target_sets[0]

    for target_set_i in range(n_target_sets):
        # Raise error if target sets have different numbers of targets across tasks
        n_target_obs = [task["Y_t"][target_set_i].size for task in tasks]
        if not all([n == n_target_obs[0] for n in n_target_obs]):
            raise ValueError(
                f"All tasks must have the same number of targets to concatenate: got {n_target_obs}. "
                "To train with Task batches containing differing numbers of targets, "
                "run the model individually over each task and average the losses."
            )

        # Raise error if target sets are different types (gridded/non-gridded) across tasks
        if isinstance(tasks[0]["X_t"][target_set_i], tuple):
            for task in tasks:
                if not isinstance(task["X_t"][target_set_i], tuple):
                    raise ValueError(
                        "All tasks must have the same type of target set (gridded or non-gridded) "
                        f"to concatenate. For target set {target_set_i}, got {type(task['X_t'][target_set_i])}."
                    )

    # For each task, store list of tuples of (x_c, y_c) (one tuple per context set)
    contexts = []
    for i, task in enumerate(tasks):
        contexts_i = list(zip(task["X_c"], task["Y_c"]))
        contexts.append(contexts_i)

    # List of tuples of merged (x_c, y_c) along batch dim with padding
    # (up to the smallest multiple of `multiple` greater than the number of contexts in each task)
    merged_context = [
        deepsensor.backend.nps.merge_contexts(
            *[context_set for context_set in contexts_i], multiple=multiple
        )
        for contexts_i in zip(*contexts)
    ]

    merged_task = copy.deepcopy(tasks[0])

    # Convert list of tuples of (x_c, y_c) to list of x_c and list of y_c
    merged_task["X_c"] = [c[0] for c in merged_context]
    merged_task["Y_c"] = [c[1] for c in merged_context]

    # This assumes that all tasks have the same number of targets
    for i in range(n_target_sets):
        if isinstance(tasks[0]["X_t"][i], tuple):
            # Target set is gridded with tuple of coords for `X_t`
            merged_task["X_t"][i] = (
                B.concat(*[t["X_t"][i][0] for t in tasks], axis=0),
                B.concat(*[t["X_t"][i][1] for t in tasks], axis=0),
            )
        else:
            # Target set is off-the-grid with tensor for `X_t`
            merged_task["X_t"][i] = B.concat(*[t["X_t"][i] for t in tasks], axis=0)
        merged_task["Y_t"][i] = B.concat(*[t["Y_t"][i] for t in tasks], axis=0)

    merged_task["time"] = [t["time"] for t in tasks]

    merged_task = Task(merged_task)

    # Apply masking
    merged_task = merged_task.mask_nans_numpy()
    merged_task = merged_task.mask_nans_nps()

    return merged_task


if __name__ == "__main__":  # pragma: no cover
    # print working directory
    import os

    print(os.path.abspath(os.getcwd()))

    import deepsensor.tensorflow as deepsensor
    from deepsensor.data.processor import DataProcessor
    from deepsensor.data.loader import TaskLoader
    from deepsensor.model.convnp import ConvNP
    from deepsensor.data.task import concat_tasks

    import xarray as xr
    import numpy as np

    da_raw = xr.tutorial.open_dataset("air_temperature")
    data_processor = DataProcessor(x1_name="lat", x2_name="lon")
    da = data_processor(da_raw)

    task_loader = TaskLoader(context=da, target=da)

    task1 = task_loader("2014-01-01", 50)
    task1["Y_c"][0][0, 0] = np.nan
    task2 = task_loader("2014-01-01", 100)

    # task1 = task1.add_batch_dim().mask_nans_numpy().mask_nans_nps()
    # task2 = task2.add_batch_dim().mask_nans_numpy().mask_nans_nps()

    merged_task = concat_tasks([task1, task2])
    print(repr(merged_task))

    print("got here")
