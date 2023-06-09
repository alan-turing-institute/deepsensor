# Load the torch extension in lab (only needs to be called once in a session)
import lab.torch as B  # noqa

# Load the TF extension in nps (to assign to deepsensor backend)
import neuralprocesses.torch as nps

import torch
import torch.nn

# Necessary for dispatching with TF and PyTorch model types when they have not yet been loaded.
# See https://beartype.github.io/plum/types.html#moduletype
from plum import clear_all_cache

clear_all_cache()

from .. import *  # noqa


def convert_to_tensor(arr):
    return torch.tensor(arr)


from deepsensor import config as deepsensor_config
from deepsensor import backend

backend.nps = nps
backend.model = torch.nn.Module
backend.convert_to_tensor = convert_to_tensor
backend.str = "torch"

B.epsilon = deepsensor_config.DEFAULT_LAB_EPSILON
