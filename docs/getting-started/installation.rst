Installation instructions
=========================

Install from `PyPI <https://pypi.org/project/deepsensor/>`_
-----------------------------------------------------------

If you want to use the latest stable release of DeepSensor and do not want/need access to the worked examples or the package's source code, we recommend installing from PyPI.

This is the easiest way to install DeepSensor.

- Install ``deepsensor``:

  .. code-block:: bash

    pip install deepsensor

- Install the backend of your choice:

  - Install ``tensorflow``:

    .. code-block:: bash

      pip install tensorflow

  - Install ``pytorch``:

    .. code-block:: bash

      pip install torch

Install from `source <https://github.com/tom-andersson/deepsensor>`_
---------------------------------------------------------------------

.. note::

    You will want to use this method if you intend on contributing to the source code of DeepSensor.

If you want to keep up with the latest changes to DeepSensor, or want/need easy access to the worked examples or the package's source code, we recommend installing from source.

This method will create a ``DeepSensor`` directory on your machine which will contain all the source code, docs and worked examples.

- Clone the repository:

  .. code-block:: bash

    git clone

- Install ``deepsensor``:

  .. code-block:: bash

    pip install -e -v .

- Install the backend of your choice:

  - Install ``tensorflow``:

    .. code-block:: bash

        pip install tensorflow

  - Install ``pytorch``:

    .. code-block:: bash

        pip install torch
