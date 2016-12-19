import numpy as np
from numpy.random import uniform

from sanitycheck.constants import *


def next_batch(bs):
    """
    Modify this function to ingest your data and returns it.
    :return: (inputs, targets). Could be a python generator.
    """
    x = np.array(uniform(size=(bs, SEQUENCE_LENGTH, 1)), dtype='float32')
    y = np.mean(x, axis=1)
    if ADD_TIME_INPUTS:
        t = np.reshape(np.tile(np.array(range(SEQUENCE_LENGTH)), (bs, 1, 1)), (bs, SEQUENCE_LENGTH, 1))
        inputs = np.squeeze(np.stack([x, t], axis=2))
    else:
        inputs = x
    return np.array(inputs, dtype='float32'), np.array(np.reshape(y, (bs, 1)), dtype='float32')
