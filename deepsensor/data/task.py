import deepsensor
import numpy as np
import lab as B
import plum
import copy


class Task(dict):
    """Task dictionary class

    Inherits from `dict` and adds methods for printing and modifying the data.
    """

    def __init__(self, task_dict: dict) -> None:
        """Initialise a Task object.

        :param task_dict: Dictionary containing the task information.
        """
        super().__init__(task_dict)

    @classmethod
    def summarise_str(cls, k, v):
        if plum.isinstance(v, B.Numeric):
            return v.shape
        elif plum.isinstance(v, tuple):
            return tuple(vi.shape for vi in v)
        elif plum.isinstance(v, list):
            return [cls.summarise_str(k, vi) for vi in v]
        else:
            return v

    @classmethod
    def summarise_repr(cls, k, v):
        if plum.isinstance(v, B.Numeric):
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

    def __str__(self):
        """Print a convenient summary of the task dictionary

        For array entries, print their shape, otherwise print the value.
        """
        s = ""
        for k, v in self.items():
            s += f"{k}: {Task.summarise_str(k, v)}\n"
        return s

    def __repr__(self):
        """Print a convenient summary of the task dictionary

        Print the type of each entry and if it is an array, print its shape, otherwise print the value.
        Print the type of each entry and if it is an array, print its shape, otherwise print the value.
        """
        s = ""
        for k, v in self.items():
            s += f"{k}: {Task.summarise_repr(k, v)}\n"
        return s

    def modify(self, f, modify_flag=None):
        """Apply function f to the array elements of a task dictionary.

        Useful for recasting to a different dtype or reshaping (e.g. adding a batch dimension).

        Parameters
        ----------
        f : function. Function to apply to the array elements of the task.
        task : dict. Task dictionary.
        modify_flag : str. Flag to set in the task dictionary's `modify` key.

        Returns
        -------
        task : dict. Task dictionary with f applied to the array elements and modify_flag set
            in the `modify` key.
        """

        def modify(k, v):
            if k in [
                "context_station_IDs",
                "target_station_IDs",
                "target_station_IDs_heldout",
            ]:
                return v
            if type(v) is list:
                return [modify(k, vi) for vi in v]
            elif type(v) is tuple:
                return (modify(k, v[0]), modify(k, v[1]))
            elif type(v) is np.ndarray:
                return f(v)
            else:
                return v  # covers metadata entries like 'region'

        self = copy.deepcopy(self)  # don't modify the original
        for k, v in self.items():
            self[k] = modify(k, v)
        self["flag"] = modify_flag

        return self  # altered by reference, but return anyway
