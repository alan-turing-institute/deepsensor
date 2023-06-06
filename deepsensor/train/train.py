import deepsensor
from deepsensor.data.task import Task
from deepsensor.data.utils import concat_tasks
from deepsensor.model.models import ConvNP

import lab as B

from typing import List


def train_epoch(
    model: ConvNP,
    tasks: List[Task],
    lr: float = 5e-5,
    batch_size: int = None,
) -> List[float]:
    """Train model for one epoch

    Args:
        model (ConvNP): Model to train
        tasks (List[Task]): List of tasks to train on
        lr (float): Learning rate
        batch_size (int, optional): Batch size. Defaults to None. If None, no batching is performed.

    Returns:
        List[float]: List of losses for each task/batch
    """
    if deepsensor.backend.str == "torch":
        # Run on GPU if available
        import torch

        if torch.cuda.is_available():
            # Set default GPU device
            torch.set_default_device("cuda")
            B.set_global_device("cuda:0")
    elif deepsensor.backend.str == "tf":
        # Run on GPU if available
        import tensorflow as tf

        if tf.test.is_gpu_available():
            # Set default GPU device
            tf.config.set_visible_devices(
                tf.config.list_physical_devices("GPU")[0], "GPU"
            )
            B.set_global_device("GPU:0")
        # Check GPU visible to tf
        # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    if deepsensor.backend.str == "tf":
        import tensorflow as tf

        opt = tf.keras.optimizers.Adam(lr)

        def train_step(tasks):
            if not isinstance(tasks, list):
                tasks = [tasks]
            with tf.GradientTape() as tape:
                task_losses = []
                for task in tasks:
                    task_losses.append(model.loss_fn(task, normalise=True))
                mean_batch_loss = B.mean(task_losses)
            grads = tape.gradient(mean_batch_loss, model.model.trainable_weights)
            opt.apply_gradients(zip(grads, model.model.trainable_weights))
            return mean_batch_loss

    elif deepsensor.backend.str == "torch":
        import torch.optim as optim

        opt = optim.Adam(model.model.parameters(), lr=lr)

        def train_step(tasks):
            if not isinstance(tasks, list):
                tasks = [tasks]
            opt.zero_grad()
            task_losses = []
            for task in tasks:
                task_losses.append(model.loss_fn(task, normalise=True))
            mean_batch_loss = B.mean(torch.stack(task_losses))
            mean_batch_loss.backward()
            opt.step()
            return mean_batch_loss.detach().cpu().numpy()

    else:
        raise NotImplementedError(f"Backend {deepsensor.backend.str} not implemented")

    if batch_size is not None:
        n_batches = len(tasks) // batch_size  # Note that this will drop the remainder
    else:
        n_batches = len(tasks)

    batch_losses = []
    for batch_i in range(n_batches):
        if batch_size is not None:
            task = concat_tasks(
                tasks[batch_i * batch_size : (batch_i + 1) * batch_size]
            )
        else:
            task = tasks[batch_i]
        batch_loss = train_step(task)
        batch_losses.append(batch_loss)

    return batch_losses
