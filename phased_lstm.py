import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import RNNCell, _linear


# Register the gradient for the mod operation. tf.mod() does not have a gradient implemented.
@ops.RegisterGradient("Mod")
def _mod_grad(op, grad):
    x, y = op.inputs
    gz = grad
    x_grad = gz
    y_grad = tf.reduce_mean(-(x // y) * gz, reduction_indices=[0], keep_dims=True)[0]
    return x_grad, y_grad


def phi(times, s, tau):
    return tf.div(tf.mod(tf.mod(times - s, tau) + tau, tau), tau)


def time_gate_fast(phase, r_on, leak_rate, training_phase, hidden_units):
    if not training_phase:
        leak_rate = 1.0
    cond_1 = tf.cast(tf.less_equal(phase, 0.5 * r_on), dtype='float32')
    cond_2 = tf.cast(tf.logical_and(tf.less(0.5 * r_on, phase), tf.less(phase, r_on)), dtype='float32')
    cond_3 = tf.cast(tf.greater_equal(phase, r_on), dtype='float32')

    term_1 = tf.mul(cond_1, 2.0 * phase / r_on)
    term_2 = tf.mul(cond_2, 2.0 - 2.0 * phase / r_on)
    term_3 = tf.mul(cond_3, leak_rate * phase)
    return term_1 + term_2 + term_3


def time_gate_slow(phase, r_on, leak_rate, training_phase, hidden_units):
    if not training_phase:
        leak_rate = 1.0
    new_phase = []
    for i in range(hidden_units):
        print('Initialize gate {}-th. (total is {}).'.format(i, hidden_units))
        new_phase.append(tf.case({tf.less(phase[i], 0.5 * r_on):
                                      lambda: 2.0 * phase[i] / r_on,
                                  tf.logical_and(tf.less(0.5 * r_on, phase[i]), tf.less(phase[i], r_on)):
                                      lambda: 2.0 - 2.0 * phase[i] / r_on},
                                 default=lambda: leak_rate * phase[i], exclusive=True))
    return tf.pack(new_phase)


class PhasedLSTMCell(RNNCell):
    def __init__(self, num_units, use_peepholes=True, training_phase=True,
                 leak_rate=0.001, activation=tanh):
        self._num_units = num_units
        self._activation = activation
        self._use_peepholes = use_peepholes
        self._leak_rate = leak_rate  # only during training
        self._training_phase = training_phase

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """ Phased long short-term memory cell (P-LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):
            # Parameters of gates are concatenated into one multiply for efficiency.
            c_prev, h_prev = state
            x = tf.reshape(inputs[:, 0], (-1, 1))
            t = inputs[:, 1][-1]  # Now we only accept one id. We have a batch so it's a bit more complex.

            # maybe the information should come from the outside. To be defined later.

            concat = _linear([x, h_prev], 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(1, 4, concat)

            tau = vs.get_variable('tau', shape=[self._num_units], dtype=inputs.dtype)
            s = vs.get_variable('s', shape=[self._num_units], dtype=inputs.dtype)
            r_on = vs.get_variable('r_on', shape=[self._num_units], dtype=inputs.dtype)

            times = tf.tile(tf.reshape(t, [-1, 1]), [1, self._num_units])
            phase = phi(times, s, tau)
            kappa = time_gate_fast(phase, r_on, self._leak_rate, self._training_phase, self._num_units)

            w_o_peephole = None
            if self._use_peepholes:
                w_i_peephole = vs.get_variable('W_I_peephole', shape=[self._num_units], dtype=inputs.dtype)
                w_f_peephole = vs.get_variable('W_F_peephole', shape=[self._num_units], dtype=inputs.dtype)
                w_o_peephole = vs.get_variable('W_O_peephole', shape=[self._num_units], dtype=inputs.dtype)
                f += w_f_peephole * c_prev
                i += w_i_peephole * c_prev

            new_c_tilde = sigmoid(f) * c_prev + sigmoid(i) * self._activation(j)
            new_c = kappa * new_c_tilde + (1 - kappa) * c_prev

            if self._use_peepholes:
                o += w_o_peephole * new_c

            new_h_tilde = sigmoid(o) * self._activation(new_c_tilde)
            new_h = kappa * new_h_tilde + (1 - kappa) * h_prev
            new_state = (new_c, new_h)
            return new_h, new_state
