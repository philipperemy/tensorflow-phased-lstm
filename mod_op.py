import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def np_mod(x, y):
    return (x % y).astype(np.float32)


def mod_grad(op, grad):
    x = op.inputs[
        0]  # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    y = op.inputs[1]  # the second argument

    return grad * 1, grad * 0  # the propagated gradient with respect to the first and second argument respectively


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1e8)))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_mod(x, y, name='Modulo'):
    with ops.op_scope([x, y], name, "mod") as name:
        z = py_func(np_mod,
                    [x, y],
                    [tf.float32],
                    name=name,
                    grad=mod_grad)  # <-- here's the call to the gradient
        z[0] = tf.reshape(z[0], [int(x.get_shape()[0])])
        return z[0]


def main():
    with tf.Session() as sess:
        x = tf.constant([0.3, 0.7, 1.2, 1.7])
        y = tf.constant([0.2, 0.5, 1.0, 2.9])
        z = tf_mod(x, y)
        x = tf.constant([0.3, 0.7, 1.2])
        y = tf.constant([0.2, 0.5, 1.0])
        z = tf_mod(x, y)
        gr = tf.gradients(z, [x, y])
        tf.initialize_all_variables().run()

        print(x.eval(), y.eval(), z.eval(), gr[0].eval(), gr[1].eval())


if __name__ == '__main__':
    main()
