from .data import *
from .model import *
from .plot import *


class Backend:
    """Backend for deepsensor

    This class is used to provide a consistent interface for either tensorflow or
    pytorch backends. It is used to assign the backend to the deepsensor module.

    Usage: blah
    """

    def __getattr__(self, attr):
        raise AttributeError(
            f"Attempting to access Backend.{attr} before {attr} has been assigned. "
            f"Please assign a backend with `import deepsensor.tensorflow` "
            f"or `import deepsensor.torch` before using backend-dependent functionality."
        )


backend = Backend()
