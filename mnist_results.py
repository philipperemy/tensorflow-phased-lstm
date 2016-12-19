import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn import dynamic_rnn

from basic_lstm import BasicLSTMCell
from helpers import FileLogger
from ml_utils import create_convolution_variable, create_bias_variable
from phased_lstm import PhasedLSTMCell


def run_lstm_mnist(lstm_cell=BasicLSTMCell, hidden_size=32, batch_size=256, steps=20):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    learning_rate = 0.001
    file_logger = FileLogger('log.tsv', ['step', 'training_loss', 'training_accuracy'])
    x = tf.placeholder('float32', [batch_size, 784, 2 if 'PhasedLSTMCell' in str(lstm_cell) else 1])
    y_ = tf.placeholder('float32', [batch_size, 10])
    initial_state = (tf.random_normal([batch_size, hidden_size], stddev=0.1),
                     tf.random_normal([batch_size, hidden_size], stddev=0.1))
    outputs, state = dynamic_rnn(lstm_cell(hidden_size), x, initial_state=initial_state, dtype=tf.float32)
    rnn_out = tf.squeeze(tf.slice(outputs, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[-1, -1, -1]))
    # _, final_hidden = state
    fc0_w = create_convolution_variable('fc0_w', [hidden_size, 10])
    fc0_b = create_bias_variable('fc0_b', [10])
    y = tf.matmul(rnn_out, fc0_w) + fc0_b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.initialize_all_variables())

    # sess.run(tf.global_variables_initializer())

    def transform_x(_x_):
        if 'PhasedLSTMCell' in str(lstm_cell):
            t = np.reshape(np.tile(np.array(range(784)), (batch_size, 1)), (batch_size, 784))
            return np.squeeze(np.stack([_x_, t], axis=2))
        t_x = np.expand_dims(_x_, axis=2)
        return t_x

    for i in range(steps):
        batch = mnist.train.next_batch(batch_size)
        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update],
                                      feed_dict={x: transform_x(batch[0]), y_: batch[1]})
        sess.run(grad_update, feed_dict={x: transform_x(batch[0]), y_: batch[1]})
        file_logger.write([i, tr_loss, tr_acc])

        # batch_test = mnist.test.next_batch(batch_size)
        # print('test accuracy %g' % sess.run(accuracy, feed_dict={x: np.expand_dims(batch_test[0], axis=2),
        #                                                y_: batch_test[
        #                                                    1]}))  # file_logger.write([step, mean_loss, benchmark_mean_loss])
    file_logger.close()


def main():
    # Vanilla LSTM
    run_lstm_mnist(lstm_cell=PhasedLSTMCell, hidden_size=32, batch_size=256, steps=2000)


if __name__ == '__main__':
    main()
