# Installation instructions

DeepSensor is a Python package that can be installed in a number of ways. In this section we will describe the two main ways to install the package.

## Install from [PyPI](https://pypi.org/project/deepsensor/)

If you want to use the latest stable release of DeepSensor and do not want/need access to the worked examples or the package's source code, we recommend installing from PyPI.

This is the easiest way to install DeepSensor.

- Install `deepsensor`:

  ```bash
  pip install deepsensor
  ```

- Install the backend of your choice:

  - Install `tensorflow`:

    ```bash
    pip install tensorflow
    ```

  - Install `pytorch`:

    ```bash
    pip install torch
    ```

## Install from [source](https://github.com/tom-andersson/deepsensor)

```{note}
You will want to use this method if you intend on contributing to the source code of DeepSensor.
```

If you want to keep up with the latest changes to DeepSensor, or want/need easy access to the worked examples or the package's source code, we recommend installing from source.

This method will create a `DeepSensor` directory on your machine which will contain all the source code, docs and worked examples.

- Clone the repository:

  ```bash
  git clone
  ```

- Install `DeepSensor`:

  ```bash
  pip install -e -v .
  ```

- Install the backend of your choice:

  - Install `tensorflow`:

    ```bash
    pip install tensorflow
    ```

  - Install `pytorch`:

    ```bash
    pip install torch
    ```
