import argparse
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import PhasedLSTMCell
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn import dynamic_rnn

from helpers.file_logger import FileLogger

num_classes = 10
mnist_img_size = 28 * 28


def run_lstm_mnist(lstm_cell=BasicLSTMCell, hidden_size=32, batch_size=256, steps=1000, log_file='log.tsv'):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    learning_rate = 0.001
    file_logger = FileLogger(log_file, ['step', 'training_loss', 'training_accuracy'])
    x_ = tf.placeholder(tf.float32, (batch_size, mnist_img_size, 1))
    t_ = tf.placeholder(tf.float32, (batch_size, mnist_img_size, 1))
    y_ = tf.placeholder(tf.float32, (batch_size, num_classes))

    if lstm_cell == PhasedLSTMCell:
        inputs = (t_, x_)
    else:
        inputs = x_
    outputs, _ = dynamic_rnn(cell=lstm_cell(hidden_size), inputs=inputs, dtype=tf.float32)
    rnn_out = tf.squeeze(outputs[:, -1, :])

    y = slim.fully_connected(inputs=rnn_out,
                             num_outputs=num_classes,
                             activation_fn=None)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    def feed_dict_phased_lstm(batch):
        img = np.expand_dims(batch[0], axis=2)
        t = np.reshape(np.tile(np.array(range(mnist_img_size)), (batch_size, 1)), (batch_size, mnist_img_size, 1))
        return {x_: img, y_: batch[1], t_: t}

    def feed_dict_basic_lstm(batch):
        img = np.expand_dims(batch[0], axis=2)
        return {x_: img, y_: batch[1]}

    for i in range(steps):
        b = mnist.train.next_batch(batch_size)
        st = time()

        if lstm_cell == PhasedLSTMCell:
            feed_dict = feed_dict_phased_lstm(b)
        else:
            feed_dict = feed_dict_basic_lstm(b)

        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update], feed_dict=feed_dict)
        print('steps = {0} | time {1:.2f} | tr_loss = {2:.3f} | tr_acc = {3:.3f}'.format(str(i).zfill(6),
                                                                                         time() - st,
                                                                                         tr_loss,
                                                                                         tr_acc))
        file_logger.write([i, tr_loss, tr_acc])

    file_logger.close()


def main():
    model_class, log_file = get_parameters()
    run_lstm_mnist(lstm_cell=model_class, hidden_size=32, batch_size=256, steps=10000, log_file=log_file)


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')  # BasicLSTMCell, PhasedLSTMCell or None
    parser.add_argument('-g', '--log_file')  # BasicLSTMCell, PhasedLSTMCell or None
    args = parser.parse_args()
    model_str = args.model
    log_file = args.log_file
    if model_str is None:
        model = PhasedLSTMCell
    else:
        model = globals()[model_str]
    print('Using model = {}'.format(model))
    return model, log_file


if __name__ == '__main__':
    main()
