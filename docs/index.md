# Welcome to DeepSensor's documentation!

DeepSensor is Python package and open-source project for modelling environmental data with neural processes.

DeepSensor aims to faithfully match the flexibility of neural processes with a simple and intuitive interface. DeepSensor wraps around the powerful [neuralprocessess package](https://github.com/wesselb/neuralprocesses) for the core modelling functionality, while allowing users to stay in the familiar [xarray](https://xarray.pydata.org/) and [pandas](https://pandas.pydata.org/) world and avoid the murky depths of tensors!

DeepSensor is also compatible with both [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/) for its machine learning abilities, thanks to the [backends package](https://github.com/wesselb/lab). Simply `import deepsensor.torch` or `import deepsensor.tensorflow` to choose between them!

```{note}
This package is currently undergoing active development. If you are interested in using DeepSensor in production, please :doc:`get in touch <contact>`.
```

## Citing DeepSensor

If you use DeepSensor in your research, please consider citing the repository. You can generate a BiBTeX entry by clicking the 'Cite this repository' button on the top right of this page.

## Quick installation

The DeepSensor package can easiest be pip installed, together with the backend of your choice. In this example we use the PyTorch backend:

```bash
pip install deepsensor torch
```

To install the TensorFlow backend instead, simply replace `torch` with `tensorflow` in the above command.

```{tableofcontents}
```
