import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from data_reader import next_batch
from helpers import FileLogger
from ml_utils import create_adam_optimizer, create_convolution_variable
from phased_lstm import PhasedLSTMCell

np.set_printoptions(threshold=np.nan)


class WaveNetTests(tf.test.TestCase):
    def test_long_sequence(self):
        with self.test_session() as sess:
            batch_size = 4
            hidden_size = 32
            learning_rate = 1e-3
            momentum = 0.9

            file_logger = FileLogger('log.tsv', ['step', 'training_loss', 'benchmark_loss'])

            x_s, y_s = next_batch(batch_size)
            x = tf.identity(x_s)
            y = tf.identity(y_s)

            lstm = PhasedLSTMCell(hidden_size)

            initial_state = (tf.random_normal([batch_size, hidden_size], stddev=0.1),
                             tf.random_normal([batch_size, hidden_size], stddev=0.1))

            outputs, state = dynamic_rnn(lstm, x, initial_state=initial_state, dtype=tf.float32)
            rnn_out = tf.squeeze(tf.slice(outputs, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[-1, -1, -1]))
            # _, final_hidden = state

            fc0_w = create_convolution_variable('fc0_w', [hidden_size, 1])
            fc0_b = tf.get_variable('fc0_b', [1])
            out = tf.nn.sigmoid(tf.matmul(rnn_out, fc0_w) + fc0_b)

            loss = tf.reduce_mean(tf.square(tf.sub(out, y)))
            optimizer = create_adam_optimizer(learning_rate, momentum)
            trainable = tf.trainable_variables()
            grad_update = optimizer.minimize(loss, var_list=trainable)

            # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            init = tf.global_variables_initializer()
            sess.run(init)

            d = collections.deque(maxlen=10)
            benchmark_d = collections.deque(maxlen=10)
            for step in range(1, int(1e9)):
                x_s, y_s = next_batch(batch_size)
                loss_value, _, pred_value = sess.run([loss, grad_update, out],
                                                     feed_dict={x: x_s, y: y_s})

                # The mean converges to 0.5 for IID U(0,1) random variables. Good benchmark.
                benchmark_d.append(np.mean(np.square(0.5 - y_s)))
                d.append(loss_value)
                mean_loss = np.mean(d)
                benchmark_mean_loss = np.mean(benchmark_d)
                file_logger.write([step, mean_loss, benchmark_mean_loss])
            file_logger.close()


if __name__ == '__main__':
    tf.test.main()
