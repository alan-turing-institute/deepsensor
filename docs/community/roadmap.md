# DeepSensor Roadmap

This page contains a list of new features that we would like to add to DeepSensor in the future.
Some of these have been raised as issues on the [GitHub issue tracker](https://github.com/tom-andersson/deepsensor/issues)
with further details.

```{note}
We will soon create a GitHub project board to track progress on these items, which will provide a more up-to-date view of the roadmap.
```

```{note}
We are unable to provide a timetable for the roadmap due to maintainer time constraints.
If you are interested in contributing to the project, check out our [](./contributing.md) page.
```

* Patch-wise training and inference
* Saving a ``TaskLoader`` when instantiated with raw xarray/pandas objects
* Non-Gaussian likelihoods
* Spatial-only modelling
* Continuous time measurements (i.e. not just discrete, uniformly sampled data on the same time grid)
* Improve forecasting functionality
* Test the framework with other models (e.g. GPs)
* Add simple baselines models (e.g. linear interpolation, GPs)
* Test and extend support for using ``dask`` in the ``DataProcessor`` and ``TaskLoader``
* Infer linked context-target sets from the ``TaskLoader`` entries, don't require user to explicitly specify ``links`` kwarg
* Improve unit test suite, increase coverage, test more edge cases, etc
