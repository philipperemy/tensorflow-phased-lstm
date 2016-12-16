import numpy as np
from numpy.random import uniform

from constants import SEQUENCE_LENGTH


def next_batch(bs):
    """
    Modify this function to ingest your data and returns it.
    :return: (inputs, targets). Could be a python generator.
    """
    x = np.array(uniform(size=(bs, SEQUENCE_LENGTH, 1)), dtype='float32')
    y = np.mean(x, axis=1)
    t = np.reshape(np.tile(np.array(range(SEQUENCE_LENGTH)), (bs, 1, 1)), (bs, SEQUENCE_LENGTH, 1))
    return np.array(x, dtype='float32'), np.array(np.reshape(y, (bs, 1)), dtype='float32'), np.array(t, dtype='float32')


if __name__ == '__main__':
    print(next_batch(4))
