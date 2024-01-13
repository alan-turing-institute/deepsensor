# Installation instructions

DeepSensor is a Python package that can be installed in a number of ways. In this section we will describe the two main ways to install the package.

## Install DeepSensor from [PyPI](https://pypi.org/project/deepsensor/)

If you want to use the latest stable release of DeepSensor and do not want/need access to the worked examples or the package's source code, we recommend installing from PyPI.

This is the easiest way to install DeepSensor.

```bash
pip install deepsensor
```

```{note}
We advise installing DeepSensor and its dependencies in a python virtual environment using a tool such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python) (other virtual environment managers are available).
```

## Install DeepSensor from [source](https://github.com/tom-andersson/deepsensor)

```{note}
You will want to use this method if you intend on contributing to the source code of DeepSensor.
```

If you want to keep up with the latest changes to DeepSensor, or want/need easy access to the worked examples or the package's source code, we recommend installing from source.

This method will create a `DeepSensor` directory on your machine which will contain all the source code, docs and worked examples.

- Clone the repository:

  ```bash
  git clone https://github.com/tom-andersson/deepsensor
  ```

- Install `DeepSensor`:

  ```bash
  pip install -v -e .
  ```
## Install PyTorch or TensorFlow

The next step, if you intend to use any of DeepSensor's deep learning modelling functionality,
is to install the deep learning backend of your choice.
Currently, DeepSensor supports PyTorch or TensorFlow.

The quickest way to install these packages is with `pip` (see below), although this doesn't guarantee
GPU functionality will work (asssuming you have a GPU).
To access GPU support, you may need to follow the installation instructions of
these libraries (PyTorch: https://pytorch.org/, TensorFlow: https://www.tensorflow.org/install).

To install `tensorflow` via pip:

```bash
pip install tensorflow
pip install tensorflow_probability
```

To install `pytorch` via pip:

```bash
pip install torch
```
