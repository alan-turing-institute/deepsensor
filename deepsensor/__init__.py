from .data import *
from .model import *
from .plot import *

class Backend:
    """Backend for deepsensor

    This class is used to provide a consistent interface for either tensorflow or
    pytorch backends. It is used to assign the backend to the deepsensor module.

    Usage: blah
    """
    pass

backend = Backend()