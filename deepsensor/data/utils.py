import copy
from typing import List

import lab as B

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
