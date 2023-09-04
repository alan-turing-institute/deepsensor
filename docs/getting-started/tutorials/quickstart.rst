====================
Tutorial: Quickstart
====================

Here we will demonstrate a simple example of training a convolutional conditional neural process (ConvCNP) to spatially interpolate ERA5 data.

We can go from imports to predictions with a trained model in less than 30 lines of code!

.. code-block:: python

    import deepsensor.torch
    from deepsensor.data.loader import TaskLoader
    from deepsensor.data.processor import DataProcessor
    from deepsensor.model.convnp import ConvNP
    from deepsensor.train.train import train_epoch

    import xarray as xr
    import pandas as pd
    import numpy as np

    # Load raw data
    ds_raw = xr.tutorial.open_dataset("air_temperature")

    # Normalise data
    data_processor = DataProcessor(x1_name="lat", x1_map=(15, 75), x2_name="lon", x2_map=(200, 330))
    ds = data_processor(ds_raw)

    # Set up task loader
    task_loader = TaskLoader(context=ds, target=ds)

    # Set up model
    model = ConvNP(data_processor, task_loader)

    # Generate training tasks with up to 10% of grid cells passed as context and all grid cells
    # passed as targets
    train_tasks = []
    for date in pd.date_range("2013-01-01", "2014-11-30")[::7]:
        task = task_loader(date, context_sampling=np.random.uniform(0.0, 0.1), target_sampling="all")
        train_tasks.append(task)

    # Train model
    for epoch in range(10):
        train_epoch(model, train_tasks, progress_bar=True)

    # Predict on new task with 10% of context data and a dense grid of target points
    test_task = task_loader("2014-12-31", 0.1)
    mean_ds, std_ds = model.predict(test_task, X_t=ds_raw)

After training, the model can predict directly to `xarray` in your data's original units and coordinate system:

.. code-block:: python

    >>> mean_ds
    <xarray.Dataset>
    Dimensions:  (time: 1, lat: 25, lon: 53)
    Coordinates:
    * time     (time) datetime64[ns] 2014-12-31
    * lat      (lat) float32 75.0 72.5 70.0 67.5 65.0 ... 25.0 22.5 20.0 17.5 15.0
    * lon      (lon) float32 200.0 202.5 205.0 207.5 ... 322.5 325.0 327.5 330.0
    Data variables:
        air      (time, lat, lon) float32 246.7 244.4 245.5 ... 290.2 289.8 289.4

We can also predict directly to `pandas` containing a timeseries of predictions at off-grid locations
by passing a `numpy` array of target locations to the `X_t` argument of `.predict`:

.. code-block:: python

    # Predict at two off-grid locations for three days in December 2014
    test_tasks = task_loader(pd.date_range("2014-12-01", "2014-12-31"), 0.1)
    mean_df, std_df = model.predict(test_tasks, X_t=np.array([[50, 280], [40, 250]]).T)

.. code-block:: python

    >>> mean_df
                                air
    time       lat  lon              
    2014-12-01 50.0 280.0  260.183056
            40.0 250.0  277.947373
    2014-12-02 50.0 280.0   261.08943
            40.0 250.0  278.219599
    2014-12-03 50.0 280.0  257.128185
            40.0 250.0  278.444229

This quickstart example is also `available as a Jupyter notebook <https://github.com/tom-andersson/deepsensor_demos/blob/main/demonstrators/quickstart.ipynb>`_ with added visualisations.
