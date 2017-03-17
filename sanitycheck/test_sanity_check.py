import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from basic_lstm import BasicLSTMCell
from helpers import FileLogger
from ml_utils import create_adam_optimizer
from ml_utils import create_weight_variable
from phased_lstm import PhasedLSTMCell
from sanitycheck.constants import *
from sanitycheck.data_reader import next_batch


def get_placeholders():
    return tf.placeholder('float32', [BATCH_SIZE, SEQUENCE_LENGTH, 2 if ADD_TIME_INPUTS else 1]), tf.placeholder(
        'float32', [BATCH_SIZE, 1])


def run_experiment(init_session=None, placeholder_def_func=get_placeholders):
    batch_size = BATCH_SIZE
    hidden_size = HIDDEN_STATES
    learning_rate = 3e-4
    momentum = 0.9

    file_logger = FileLogger('log.tsv', ['step', 'training_loss', 'benchmark_loss'])

    x, y = placeholder_def_func()
    if ADD_TIME_INPUTS:
        lstm = PhasedLSTMCell(hidden_size)
        print('Using PhasedLSTMCell impl.')
    else:
        lstm = BasicLSTMCell(hidden_size)
        print('Using BasicLSTMCell impl.')

    initial_state = (tf.random_normal([batch_size, hidden_size], stddev=0.1),
                     tf.random_normal([batch_size, hidden_size], stddev=0.1))

    outputs, state = dynamic_rnn(lstm, x, initial_state=initial_state, dtype=tf.float32)
    rnn_out = tf.squeeze(tf.slice(outputs, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[-1, -1, -1]))
    # _, final_hidden = state

    fc0_w = create_weight_variable('fc0_w', [hidden_size, 1])
    fc0_b = tf.get_variable('fc0_b', [1])
    out = tf.matmul(rnn_out, fc0_w) + fc0_b

    loss = tf.reduce_mean(tf.square(tf.sub(out, y)))
    optimizer = create_adam_optimizer(learning_rate, momentum)
    trainable = tf.trainable_variables()
    grad_update = optimizer.minimize(loss, var_list=trainable)

    if init_session is not None:
        sess = init_session
    else:
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # lstm.__call__(x[:, 0, :], initial_state, scope=None)

    d = collections.deque(maxlen=10)
    benchmark_d = collections.deque(maxlen=10)
    for step in range(1, int(1e9)):
        x_s, y_s = next_batch(batch_size)
        loss_value, _, pred_value = sess.run([loss, grad_update, out], feed_dict={x: x_s, y: y_s})
        # The mean converges to 0.5 for IID U(0,1) random variables. Good benchmark.
        benchmark_d.append(np.mean(np.square(0.5 - y_s)))
        d.append(loss_value)
        mean_loss = np.mean(d)
        benchmark_mean_loss = np.mean(benchmark_d)
        file_logger.write([step, mean_loss, benchmark_mean_loss])
    file_logger.close()


if __name__ == '__main__':
    run_experiment()
