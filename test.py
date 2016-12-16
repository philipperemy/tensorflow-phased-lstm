import numpy as np
import tensorflow as tf

from constants import BATCH_SIZE
from data_reader import next_batch
from main import main

np.set_printoptions(threshold=np.nan)

SESSION = None


def get_placeholders_test():
    np.random.seed(123)
    x_s, y_s = next_batch(BATCH_SIZE)
    x = tf.identity(x_s)
    y = tf.identity(y_s)
    return x, y


class PhasedLSTMTests(tf.test.TestCase):
    def test_1(self):
        global SESSION
        with self.test_session() as sess:
            SESSION = sess
            main(sess, get_placeholders_test)
            # debug a forward call to the RNN.


if __name__ == '__main__':
    tf.test.main()
