import argparse
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn import dynamic_rnn

from tensorflow.contrib.rnn import BasicLSTMCell
from helpers import FileLogger
from ml_utils import create_weight_variable, create_bias_variable
from phased_lstm import PhasedLSTMCell


def run_lstm_mnist(lstm_cell=BasicLSTMCell, hidden_size=32, batch_size=256, steps=1000):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    learning_rate = 0.001
    file_logger = FileLogger('log.tsv', ['step', 'training_loss', 'training_accuracy'])
    x = tf.placeholder('float32', [batch_size, 784, 2 if lstm_cell == PhasedLSTMCell else 1])
    y_ = tf.placeholder('float32', [batch_size, 10])
    initial_states = (tf.random_normal([batch_size, hidden_size], stddev=0.1),
                      tf.random_normal([batch_size, hidden_size], stddev=0.1))
    outputs, _ = dynamic_rnn(lstm_cell(hidden_size), x, initial_state=initial_states, dtype=tf.float32)
    rnn_out = tf.squeeze(outputs[:, -1, :])

    fc0_w = create_weight_variable('fc0_w', [hidden_size, 10])
    fc0_b = create_bias_variable('fc0_b', [10])
    y = tf.matmul(rnn_out, fc0_w) + fc0_b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    def transform_x(_x_):
        if lstm_cell == PhasedLSTMCell:
            t = np.reshape(np.tile(np.array(range(784)), (batch_size, 1)), (batch_size, 784))
            return np.squeeze(np.stack([_x_, t], axis=2))
        t_x = np.expand_dims(_x_, axis=2)
        return t_x

    for i in range(steps):
        batch = mnist.train.next_batch(batch_size)
        st = time()
        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update],
                                      feed_dict={x: transform_x(batch[0]), y_: batch[1]})
        print('Forward-Backward pass took {0:.2f}s to complete.'.format(time() - st))
        file_logger.write([i, tr_loss, tr_acc])

    file_logger.close()


def main():
    model_class = get_model_class()
    run_lstm_mnist(lstm_cell=model_class, hidden_size=32, batch_size=256, steps=10000)


def get_model_class():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')  # BasicLSTMCell, PhasedLSTMCell or None
    args = parser.parse_args()
    model_str = args.model
    if model_str is None:
        model = PhasedLSTMCell
    else:
        model = globals()[model_str]
    print('Using model = {}'.format(model))
    return model


if __name__ == '__main__':
    main()
