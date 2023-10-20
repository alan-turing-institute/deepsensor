# DeepSensor Roadmap

```{note}
We will soon create a GitHub project board to track progress on these items, which will provide a more up-to-date view of the roadmap.
```

```{note}
We are unable to provide a timetable for the roadmap due to maintainer time constraints. If you are interested in contributing to the project, check out :doc:`Contributing Guide <contributing>`.
```

* Improve documentation
* Support patch-wise training and inference
* Support non-Gaussian likelihoods
* Support spatial-only modelling
* Support or explore supporting continuous time measurements (i.e. not just discrete, uniformly sampled data on the same time grid)
* Improve forecasting functionality
* Test the framework with other models (e.g. GPs)
* Add simple baselines models (e.g. linear interpolation, GPs)
* Test and extend support for using ``dask`` in the ``DataProcessor`` and ``TaskLoader``
* Infer linked context-target sets from the ``TaskLoader`` entries, don't require user to explicitly specify ``links`` kwarg
* Improve unit test suite, increase coverage, test more edge cases, etc
