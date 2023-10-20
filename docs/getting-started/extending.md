# Extending DeepSensor with new models

To extend DeepSensor with a new model, simply create a new class that inherits from `deepsensor.model.DeepSensorModel` and implement the low-level prediction methods defined in `deepsensor.model.ProbabilisticModel`, such as `.mean` and `.stddev`.

In this example, we'll create a new model called `ExampleModel`:

```python
class ExampleModel(DeepSensorModel):
    """
    A very naive model that predicts the mean of the first context set
    with a fixed stddev.
    """

    def __init__(self, data_processor: DataProcessor, task_loader: TaskLoader):
        # Initiate the parent class (DeepSensorModel) with the
        # provided data processor and task loader:
        super().__init__(data_processor, task_loader)

    def mean(self, task: Task):
        """Compute mean at target locations"""
        return np.mean(task["Y_c"][0])

    def stddev(self, task: Task):
        """Compute stddev at target locations"""
        return 0.1

    ...
```

After creating `ExampleModel` in this way, it can be used in the same way as the built-in :class:`~deepsensor.model.convnp.ConvNP` model.

See [this Jupyter notebook](https://github.com/tom-andersson/deepsensor_gallery/blob/main/demonstrators/extending_models.ipynb) for more details.
