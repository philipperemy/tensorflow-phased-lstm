import numpy as np
from numpy.random import uniform


def next_batch(bs):
    """
    Modify this function to ingest your data and returns it.
    :return: (inputs, targets). Could be a python generator.
    """
    x = np.array(uniform(size=(bs, 16, 1)), dtype='float32')
    y = np.mean(x, axis=1)
    return np.array(x, dtype='float32'), np.array(np.reshape(y, (bs, 1, 1)), dtype='float32')


if __name__ == '__main__':
    print(next_batch(4))
