import deepsensor
from deepsensor.data.task import Task
from deepsensor.model.convnp import ConvNP, concat_tasks

import numpy as np

import lab as B

from typing import List


def set_gpu_default_device():
    if deepsensor.backend.str == "torch":
        # Run on GPU if available
        import torch

        if torch.cuda.is_available():
            # Set default GPU device
            torch.set_default_device("cuda")
            B.set_global_device("cuda:0")
        else:
            raise RuntimeError("No GPU available: torch.cuda.is_available() == False")
    elif deepsensor.backend.str == "tf":
        # Run on GPU if available
        import tensorflow as tf

        if tf.test.is_gpu_available():
            # Set default GPU device
            tf.config.set_visible_devices(
                tf.config.list_physical_devices("GPU")[0], "GPU"
            )
            B.set_global_device("GPU:0")
        else:
            raise RuntimeError("No GPU available: tf.test.is_gpu_available() == False")

    else:
        raise NotImplementedError(f"Backend {deepsensor.backend.str} not implemented")


def train_epoch(
    model: ConvNP,
    tasks: List[Task],
    lr: float = 5e-5,
    batch_size: int = None,
    opt=None,
    progress_bar=False,
    tqdm_notebook=False,
) -> List[float]:
    """Train model for one epoch

    Args:
        model (ConvNP): Model to train
        tasks (List[Task]): List of tasks to train on
        lr (float): Learning rate
        batch_size (int, optional): Batch size. Defaults to None. If None, no batching is performed.
        opt: TF or Torch optimiser object. Optional.

    Returns:
        List[float]: List of losses for each task/batch
    """
    if deepsensor.backend.str == "tf":
        import tensorflow as tf

        if opt is None:
            opt = tf.keras.optimizers.Adam(lr)

        def train_step(tasks):
            if not isinstance(tasks, list):
                tasks = [tasks]
            with tf.GradientTape() as tape:
                task_losses = []
                for task in tasks:
                    task_losses.append(model.loss_fn(task, normalise=True))
                mean_batch_loss = B.mean(B.stack(*task_losses))
            grads = tape.gradient(mean_batch_loss, model.model.trainable_weights)
            opt.apply_gradients(zip(grads, model.model.trainable_weights))
            return mean_batch_loss

    elif deepsensor.backend.str == "torch":
        import torch.optim as optim

        if opt is None:
            opt = optim.Adam(model.model.parameters(), lr=lr)

        def train_step(tasks):
            if not isinstance(tasks, list):
                tasks = [tasks]
            opt.zero_grad()
            task_losses = []
            for task in tasks:
                task_losses.append(model.loss_fn(task, normalise=True))
            mean_batch_loss = B.mean(B.stack(*task_losses))
            mean_batch_loss.backward()
            opt.step()
            return mean_batch_loss.detach().cpu().numpy()

    else:
        raise NotImplementedError(f"Backend {deepsensor.backend.str} not implemented")

    tasks = np.random.permutation(tasks)

    if batch_size is not None:
        n_batches = len(tasks) // batch_size  # Note that this will drop the remainder
    else:
        n_batches = len(tasks)

    if tqdm_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    batch_losses = []
    for batch_i in tqdm(range(n_batches), disable=not progress_bar):
        if batch_size is not None:
            task = concat_tasks(
                tasks[batch_i * batch_size : (batch_i + 1) * batch_size]
            )
        else:
            task = tasks[batch_i]
        batch_loss = train_step(task)
        batch_losses.append(batch_loss)

    return batch_losses
