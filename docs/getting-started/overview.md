# Overview: Why DeepSensor?

Machine learning (ML) has now made its way from the fringes to the
frontiers of environmental science. However, many of the success stories so far have used
gridded reanalysis data as the target variables.
There are growing calls for more flexible ML approaches that can handle the challenges of environmental observations
to tackle a range of
prediction tasks like forecasting, downscaling, satellite gap-filling, and sensor placement (i.e.
telling us where to put sensors to get the most information about the environment).

![DeepSensor applications](../../figs/deepsensor_application_examples.png)

Environmental data is challenging for conventional ML architectures because
as it can be multi-modal, multi-resolution, and have missing data.
Differing data modalities provide different information:
* station data provides high quality localised information but may not represent surroundings;
* satellite data provides huge areas of high-res information, but only indirectly sense target quantities and can have missing data;
* reanalysis data provide a spatiotemporally complete picture of the atmosphere and oceans but are limited by model bias and coarse resolution.

Neural processes have emerged as promising ML architectures for environmental data because they can:
* efficiently fuse multi-modal and multi-resolution data,
* handle missing observations,
* capture prediction uncertainty.

The DeepSensor Python package streamlines the application of NPs
to environmental sciences by plugging together the `xarray`, `pandas`, and `neuralprocesses` packages with a user-friendly interface that
enables rapid experimentation.
The pacakge allows users to tackle diverse environmental modelling tasks,
such as sensor placement, forecasting, downscaling, and satellite gap-filling.

DeepSensor aims to:
* Drastically reduces effort to apply NPs to environmental prediction tasks, allowing DeepSensor users to focus on the science 
* Accelerate research by building an open-source software community
* Generate a positive feedback loop between research and software
* Accelerate the next generation of environmental ML 

If this interests you, then let's get started!