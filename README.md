[//]: # (![]&#40;figs/DeepSensorLogo.png&#41;)
<p style="text-align: center;">
<img src="figs/DeepSensorLogo.png" width="800", margin: -150px 0px 0px -150px; />
</p>

<ul style="margin-top:-50px;">


<p style="text-align: center;">A Python package for modelling environmental data with processes</p>
-----------

[![release](https://img.shields.io/badge/release-v0.1.3-green?logo=github)](https://github.com/tom-andersson/deepsensor/releases)
![Tests](https://github.com/tom-andersson/deepsensor/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/tom-andersson/deepsensor/badge.svg?branch=main)](https://coveralls.io/github/tom-andersson/deepsensor?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


**DeepSensor** is an open source project and Python package
for modelling xarray and pandas data with neural processes (NPs).

**NOTE**: This package is currently undergoing very active development. If you are interested in using
DeepSensor, please get in touch first (tomand@bas.ac.uk).

Why neural processes?
-----------
NPs are a highly flexible class of probabilistic models that can:
- ingest multiple context sets (i.e. data streams) containing gridded or pointwise observations
- handle multiple gridded resolutions
- predict at arbitrary target locations
- quantify prediction uncertainty

These capabilities make NPs well suited to modelling spatio-temporal data, such as
satellite observations, climate model output, and in-situ measurements.
NPs have been used for range of environmental applications, including:
- spatial interpolation (sensor placement)
- downscaling (i.e. super-resolution)
- forecasting

Why DeepSensor?
-----------
DeepSensor aims to faithfully match the flexibility of NPs with a simple and intuitive
interface.
DeepSensor wraps around the powerful [neuralprocessess](https://github.com/wesselb/neuralprocesses)
package for the core modelling functionality, while allowing users to stay in
the familiar [xarray](https://xarray.pydata.org) and [pandas](https://pandas.pydata.org) world
and avoid the murky depths of tensors!

Backend agnosticism
-----------
DeepSensor leverages the [backends](https://github.com/wesselb/lab) package to be compatible with
either [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/)
deep learning libraries.
Simply `import deepsensor.torch` or `import deepsensor.tensorflow` to choose your backend!

Quick start
----------

Here we will demonstrate a simple example of training a convolutional conditional neural process
(ConvCNP) to spatially interpolate ERA5 data.
First, pip install the package. In this case we will use the TensorFlow backend.
```bash
pip install deepsensor
pip install tensorflow
pip install tensorflow-probability
```

We can go from imports to predictions with a trained model in <30 lines of code!

```python
import deepsensor.tensorflow
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
for date in pd.date_range("2013-01-01", "2014-11-30"):
    task = task_loader(date, context_sampling=np.random.uniform(0.0, 0.1), target_sampling="all")
    train_tasks.append(task)

# Train model
for epoch in range(100):
    train_epoch(model, train_tasks, progress_bar=True)

# Predict on new task with 10% of context data
test_task = task_loader("2014-12-31", 0.1)
mean_ds, std_ds = model.predict(test_task, X_t=ds_raw)
```

After training, the model can predict directly to `xarray` in your data's original units and coordinate system:
```python
>>> mean_ds
<xarray.Dataset>
Dimensions:  (time: 1, lat: 25, lon: 53)
Coordinates:
  * time     (time) datetime64[ns] 2014-12-31
  * lat      (lat) float32 75.0 72.5 70.0 67.5 65.0 ... 25.0 22.5 20.0 17.5 15.0
  * lon      (lon) float32 200.0 202.5 205.0 207.5 ... 322.5 325.0 327.5 330.0
Data variables:
    air      (time, lat, lon) float32 246.7 244.4 245.5 ... 290.2 289.8 289.4
```

Learn more
----------
- Documentation: TODO
- Issue tracker: https://github.com/tom-andersson/deepsensor/issues
- Source code: https://github.com/tom-andersson/deepsensor

**Talks**
- Environmental Sensor Placement with ConvGNPs: https://youtu.be/v0pmqh09u1Y

**Papers**
- TODO
