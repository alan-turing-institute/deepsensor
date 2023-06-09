# Load the tensorflow extension in lab (only needs to be called once in a session)
import lab.tensorflow as B  # noqa

# Load the TF extension in nps (to assign to deepsensor backend)
import neuralprocesses.tensorflow as nps

import tensorflow as tf
import tensorflow.keras

# Necessary for dispatching with TF and PyTorch model types when they have not yet been loaded.
# See https://beartype.github.io/plum/types.html#moduletype
from plum import clear_all_cache

clear_all_cache()

from .. import *  # noqa


def convert_to_tensor(arr):
    return tf.convert_to_tensor(arr)


from deepsensor import config as deepsensor_config
from deepsensor import backend

backend.nps = nps
backend.model = tf.keras.Model
backend.convert_to_tensor = convert_to_tensor
backend.str = "tf"

B.epsilon = deepsensor_config.DEFAULT_LAB_EPSILON
