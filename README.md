[//]: # (![]&#40;figs/DeepSensorLogo.png&#41;)
<ul style="text-align: center;">
<img src="https://raw.githubusercontent.com/alan-turing-institute/deepsensor/main/figs/DeepSensorLogo2.png" width="700"/>
</ul>

<ul style="margin-top:0px;">


<p style="text-align: center; font-size: 15px">A Python package and open-source project for modelling environmental
data with neural processes</p>

-----------

[![release](https://img.shields.io/badge/release-v0.3.6-green?logo=github)](https://github.com/alan-turing-institute/deepsensor/releases)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://alan-turing-institute.github.io/deepsensor/)
![Tests](https://github.com/alan-turing-institute/deepsensor/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/alan-turing-institute/deepsensor/badge.svg?branch=main)](https://coveralls.io/github/alan-turing-institute/deepsensor?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![slack](https://img.shields.io/badge/slack-deepsensor-purple.svg?logo=slack)](https://ai4environment.slack.com/archives/C05NQ76L87R)
[![All Contributors](https://img.shields.io/github/all-contributors/alan-turing-institute/deepsensor?color=ee8449&style=flat-square)](#contributors)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/alan-turing-institute/deepsensor/blob/main/LICENSE)

DeepSensor streamlines the application of neural processes (NPs) to environmental sciences by
providing a simple interface for building, training, and evaluating NPs using `xarray` and `pandas`
data. Our developers and users form an open-source community whose vision is to accelerate the next
generation of environmental ML research. The DeepSensor Python package facilitates this by
drastically reducing the time and effort required to apply NPs to environmental prediction tasks.
This allows DeepSensor users to focus on the science and rapidly iterate on ideas.

DeepSensor is an experimental package, and we
welcome [contributions from the community](https://github.com/alan-turing-institute/deepsensor/blob/main/CONTRIBUTING.md).
We have an active Slack channel for code and research discussions; you can join by [signing up for the Turing Environment & Sustainability stakeholder community](https://forms.office.com/pages/responsepage.aspx?id=p_SVQ1XklU-Knx-672OE-ZmEJNLHTHVFkqQ97AaCfn9UMTZKT1IwTVhJRE82UjUzMVE2MThSOU5RMC4u). The form includes a question on signing up for the Slack team, where you can find DeepSensor's channel.

![DeepSensor example application figures](https://raw.githubusercontent.com/alan-turing-institute/deepsensor/main/figs/deepsensor_application_examples.png)

Why neural processes?
-----------
NPs are a highly flexible class of probabilistic models that offer unique opportunities to model
satellite observations, climate model output, and in-situ measurements.
Their key features are the ability to:

- ingest multiple data streams of pointwise or gridded modalities
- handle missing data and varying resolutions
- predict at arbitrary target locations
- quantify prediction uncertainty

These capabilities make NPs well suited to a range of
spatio-temporal data fusion tasks such as downscaling, sensor placement, gap-filling, and forecasting.

Why DeepSensor?
-----------
This package aims to faithfully match the flexibility of NPs with a simple and intuitive interface.
Under the hood, DeepSensor wraps around the
powerful [neuralprocessess](https://github.com/wesselb/neuralprocesses) package for core modelling
functionality, while allowing users to stay in the familiar [xarray](https://xarray.pydata.org)
and [pandas](https://pandas.pydata.org) world from end-to-end.
DeepSensor also provides convenient plotting tools and active learning functionality for finding
optimal [sensor placements](https://doi.org/10.1017/eds.2023.22).

Documentation
-----------
We have an extensive documentation page [here](https://alan-turing-institute.github.io/deepsensor/),
containing steps for getting started, a user guide built from reproducible Jupyter notebooks,
learning resources, research ideas, community information, an API reference, and more!

DeepSensor Gallery
-----------
For real-world DeepSensor research demonstrators, check out the
[DeepSensor Gallery](https://github.com/tom-andersson/deepsensor_gallery).
Consider submitting a notebook showcasing your research!

Deep learning library agnosticism
-----------
DeepSensor leverages the [backends](https://github.com/wesselb/lab) package to be compatible with
either [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).
Simply `import deepsensor.torch` or `import deepsensor.tensorflow` to choose between them!

Quick start
----------

Here we will demonstrate a simple example of training a convolutional conditional neural process
(ConvCNP) to spatially interpolate random grid cells of NCEP reanalysis air temperature data
over the US. First, pip install the package. In this case we will use the PyTorch backend
(note: follow the [PyTorch installation instructions](https://pytorch.org/) if you
want GPU support).

```bash
pip install deepsensor
pip install torch
```

We can go from imports to predictions with a trained model in less than 30 lines of code!

```python
import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer

import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load raw data
ds_raw = xr.tutorial.open_dataset("air_temperature")

# Normalise data
data_processor = DataProcessor(x1_name="lat", x2_name="lon")
ds = data_processor(ds_raw)

# Set up task loader
task_loader = TaskLoader(context=ds, target=ds)

# Set up model
model = ConvNP(data_processor, task_loader)

# Generate training tasks with up 100 grid cells as context and all grid cells
#   as targets
train_tasks = []
for date in pd.date_range("2013-01-01", "2014-11-30")[::7]:
    N_context = np.random.randint(0, 100)
    task = task_loader(date, context_sampling=N_context, target_sampling="all")
    train_tasks.append(task)

# Train model
trainer = Trainer(model, lr=5e-5)
for epoch in tqdm(range(10)):
    batch_losses = trainer(train_tasks)

# Predict on new task with 50 context points and a dense grid of target points
test_task = task_loader("2014-12-31", context_sampling=50)
pred = model.predict(test_task, X_t=ds_raw)
```

After training, the model can predict directly to `xarray` in your data's original units and
coordinate system:

```python
>>> pred["air"]
<xarray.Dataset>
Dimensions:  (time: 1, lat: 25, lon: 53)
Coordinates:
  * time     (time) datetime64[ns] 2014-12-31
  * lat      (lat) float32 75.0 72.5 70.0 67.5 65.0 ... 25.0 22.5 20.0 17.5 15.0
  * lon      (lon) float32 200.0 202.5 205.0 207.5 ... 322.5 325.0 327.5 330.0
Data variables:
    mean     (time, lat, lon) float32 267.7 267.2 266.4 ... 297.5 297.8 297.9
    std      (time, lat, lon) float32 9.855 9.845 9.848 ... 1.356 1.36 1.487
```

We can also predict directly to `pandas` containing a timeseries of predictions at off-grid
locations
by passing a `numpy` array of target locations to the `X_t` argument of `.predict`:

```python
# Predict at two off-grid locations over December 2014 with 50 random, fixed context points
test_tasks = task_loader(pd.date_range("2014-12-01", "2014-12-31"), 50, seed_override=42)
pred = model.predict(test_tasks, X_t=np.array([[50, 280], [40, 250]]).T)
```

```python
>>> pred["air"]
                          mean       std
time       lat lon                      
2014-12-01 50  280  260.282562  5.743976
           40  250  270.770111  4.271546
2014-12-02 50  280  255.572098  6.165956
           40  250  277.588745  3.727404
2014-12-03 50  280  260.894196   6.02924
...                        ...       ...
2014-12-29 40  250  266.594421  4.268469
2014-12-30 50  280  250.936386  7.048379
           40  250  262.225464  4.662592
2014-12-31 50  280  249.397919  7.167142
           40  250  257.955505  4.697775

[62 rows x 2 columns]
```

DeepSensor offers far more functionality than this simple example demonstrates.
For more information on the package's capabilities, check out the
[User Guide](https://alan-turing-institute.github.io/deepsensor/user-guide/index.html)
in the documentation.

## Citing DeepSensor

If you use DeepSensor in your research, please consider citing this repository.
You can generate a BiBTeX entry by clicking the 'Cite this repository' button
on the top right of this page.

## Funding

DeepSensor is funded by [The Alan Turing Institute](https://www.turing.ac.uk/) under the [Environmental monitoring: blending satellite and surface data](https://www.turing.ac.uk/research/research-projects/environmental-monitoring-blending-satellite-and-surface-data) and [Scivision](https://www.turing.ac.uk/research/research-projects/scivision) projects, led by PI [Dr Scott Hosking](https://www.turing.ac.uk/people/researchers/scott-hosking).

## Contributors

We appreciate all contributions to DeepSensor, big or small, code-related or not, and we thank all
contributors below for supporting open-source software and research.
For code-specific contributions, check out our graph of [code contributions](https://github.com/alan-turing-institute/deepsensor/graphs/contributors).
See our [contribution guidelines](https://github.com/alan-turing-institute/deepsensor/blob/main/CONTRIBUTING.md)
if you would like to join this list!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/acocac"><img src="https://avatars.githubusercontent.com/u/13321552?v=4?s=100" width="100px;" alt="Alejandro ©"/><br /><sub><b>Alejandro ©</b></sub></a><br /><a href="#userTesting-acocac" title="User Testing">📓</a> <a href="#bug-acocac" title="Bug reports">🐛</a> <a href="#mentoring-acocac" title="Mentoring">🧑‍🏫</a> <a href="#ideas-acocac" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-acocac" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/annavaughan"><img src="https://avatars.githubusercontent.com/u/45528489?v=4?s=100" width="100px;" alt="Anna Vaughan"/><br /><sub><b>Anna Vaughan</b></sub></a><br /><a href="#research-annavaughan" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://davidwilby.dev"><img src="https://avatars.githubusercontent.com/u/24752124?v=4?s=100" width="100px;" alt="David Wilby"/><br /><sub><b>David Wilby</b></sub></a><br /><a href="#doc-davidwilby" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://inconsistentrecords.co.uk"><img src="https://avatars.githubusercontent.com/u/731727?v=4?s=100" width="100px;" alt="Jim Circadian"/><br /><sub><b>Jim Circadian</b></sub></a><br /><a href="#ideas-JimCircadian" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-JimCircadian" title="Project Management">📆</a> <a href="#maintenance-JimCircadian" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jonas-scholz123"><img src="https://avatars.githubusercontent.com/u/37850411?v=4?s=100" width="100px;" alt="Jonas Scholz"/><br /><sub><b>Jonas Scholz</b></sub></a><br /><a href="#userTesting-jonas-scholz123" title="User Testing">📓</a> <a href="#research-jonas-scholz123" title="Research">🔬</a> <a href="#code-jonas-scholz123" title="Code">💻</a> <a href="#bug-jonas-scholz123" title="Bug reports">🐛</a> <a href="#ideas-jonas-scholz123" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.westerling.nu"><img src="https://avatars.githubusercontent.com/u/7298727?v=4?s=100" width="100px;" alt="Kalle Westerling"/><br /><sub><b>Kalle Westerling</b></sub></a><br /><a href="#doc-kallewesterling" title="Documentation">📖</a> <a href="#infra-kallewesterling" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-kallewesterling" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-kallewesterling" title="Project Management">📆</a> <a href="#promotion-kallewesterling" title="Promotion">📣</a> <a href="#question-kallewesterling" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://kenzaxtazi.github.io"><img src="https://avatars.githubusercontent.com/u/43008274?v=4?s=100" width="100px;" alt="Kenza Tazi"/><br /><sub><b>Kenza Tazi</b></sub></a><br /><a href="#ideas-kenzaxtazi" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://magnusross.github.io/about"><img src="https://avatars.githubusercontent.com/u/51709759?v=4?s=100" width="100px;" alt="Magnus Ross"/><br /><sub><b>Magnus Ross</b></sub></a><br /><a href="#tutorial-magnusross" title="Tutorials">✅</a> <a href="#data-magnusross" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://nilsleh.info/"><img src="https://avatars.githubusercontent.com/u/35272119?v=4?s=100" width="100px;" alt="Nils Lehmann"/><br /><sub><b>Nils Lehmann</b></sub></a><br /><a href="#ideas-nilsleh" title="Ideas, Planning, & Feedback">🤔</a> <a href="#userTesting-nilsleh" title="User Testing">📓</a> <a href="#bug-nilsleh" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/polpel"><img src="https://avatars.githubusercontent.com/u/56694450?v=4?s=100" width="100px;" alt="Paolo Pelucchi"/><br /><sub><b>Paolo Pelucchi</b></sub></a><br /><a href="#userTesting-polpel" title="User Testing">📓</a> <a href="#bug-polpel" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://rohitrathore.netlify.app/"><img src="https://avatars.githubusercontent.com/u/42641738?v=4?s=100" width="100px;" alt="Rohit Singh Rathaur"/><br /><sub><b>Rohit Singh Rathaur</b></sub></a><br /><a href="#code-RohitRathore1" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://scotthosking.com"><img src="https://avatars.githubusercontent.com/u/10783052?v=4?s=100" width="100px;" alt="Scott Hosking"/><br /><sub><b>Scott Hosking</b></sub></a><br /><a href="#fundingFinding-scotthosking" title="Funding Finding">🔍</a> <a href="#ideas-scotthosking" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-scotthosking" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.bas.ac.uk/profile/tomand"><img src="https://avatars.githubusercontent.com/u/26459412?v=4?s=100" width="100px;" alt="Tom Andersson"/><br /><sub><b>Tom Andersson</b></sub></a><br /><a href="#code-tom-andersson" title="Code">💻</a> <a href="#research-tom-andersson" title="Research">🔬</a> <a href="#maintenance-tom-andersson" title="Maintenance">🚧</a> <a href="#bug-tom-andersson" title="Bug reports">🐛</a> <a href="#test-tom-andersson" title="Tests">⚠️</a> <a href="#tutorial-tom-andersson" title="Tutorials">✅</a> <a href="#doc-tom-andersson" title="Documentation">📖</a> <a href="#review-tom-andersson" title="Reviewed Pull Requests">👀</a> <a href="#talk-tom-andersson" title="Talks">📢</a> <a href="#question-tom-andersson" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://wessel.ai"><img src="https://avatars.githubusercontent.com/u/1444448?v=4?s=100" width="100px;" alt="Wessel"/><br /><sub><b>Wessel</b></sub></a><br /><a href="#research-wesselb" title="Research">🔬</a> <a href="#code-wesselb" title="Code">💻</a> <a href="#ideas-wesselb" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://patel-zeel.github.io"><img src="https://avatars.githubusercontent.com/u/59758528?v=4?s=100" width="100px;" alt="Zeel B Patel"/><br /><sub><b>Zeel B Patel</b></sub></a><br /><a href="#bug-patel-zeel" title="Bug reports">🐛</a> <a href="#code-patel-zeel" title="Code">💻</a> <a href="#userTesting-patel-zeel" title="User Testing">📓</a> <a href="#ideas-patel-zeel" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ots22"><img src="https://avatars.githubusercontent.com/u/5434836?v=4?s=100" width="100px;" alt="ots22"/><br /><sub><b>ots22</b></sub></a><br /><a href="#ideas-ots22" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
