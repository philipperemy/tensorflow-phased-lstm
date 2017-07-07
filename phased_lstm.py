import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

_linear = rnn_cell_impl._linear


def random_exp_initializer(minval=0, maxval=None, seed=None,
                           dtype=dtypes.float32):
    '''Returns an initializer that generates tensors with an exponential distribution.
    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type.
    Returns:
      An initializer that generates tensors with an exponential distribution.
    '''

    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.exp(random_ops.random_uniform(shape, minval, maxval, dtype, seed=seed))

    return _initializer


# Register the gradient for the mod operation. tf.mod() does not have a gradient implemented.
@ops.RegisterGradient('FloorMod')
def _mod_grad(op, grad):
    x, y = op.inputs
    gz = grad
    x_grad = gz
    y_grad = tf.reduce_mean(-(x // y) * gz, axis=[0], keep_dims=True)[0]
    return x_grad, y_grad


def phi(times, s, tau):
    # return tf.div(tf.mod(tf.mod(times - s, tau) + tau, tau), tau)
    return tf.div(tf.mod(times - s, tau), tau)


def time_gate_fast_2(phase, r_on, leak_rate, training_phase):
    if not training_phase:
        leak_rate = 1.0
    is_up = tf.less(phase, (r_on * 0.5))
    is_down = tf.logical_and(tf.less(phase, r_on), tf.logical_not(is_up))
    time_gate = tf.where(is_up, 2 * phase / r_on, tf.where(is_down, 2. - 2. * (phase / r_on), leak_rate * phase))
    return time_gate


def time_gate_fast(phase, r_on, leak_rate, training_phase):
    if not training_phase:
        leak_rate = 1.0
    cond_1 = tf.cast(tf.less_equal(phase, 0.5 * r_on), dtype='float32')
    cond_2 = tf.cast(tf.logical_and(tf.less(0.5 * r_on, phase), tf.less(phase, r_on)), dtype='float32')
    cond_3 = tf.cast(tf.greater_equal(phase, r_on), dtype='float32')

    term_1 = tf.multiply(cond_1, 2.0 * phase / r_on)
    term_2 = tf.multiply(cond_2, 2.0 - 2.0 * phase / r_on)
    term_3 = tf.multiply(cond_3, leak_rate * phase)
    return term_1 + term_2 + term_3


class PhasedLSTMCell(RNNCell):
    def __init__(self, num_units, use_peepholes=True, training_phase=True,
                 leak_rate=0.001, r_on_init=0.05, tau_init=6., activation=tanh):
        self._num_units = num_units
        self._activation = activation
        self._use_peepholes = use_peepholes
        self._leak_rate = leak_rate  # only during training
        self._training_phase = training_phase
        self.r_on_init = r_on_init
        self.tau_init = tau_init

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

            # (batch_size, seq_len, 2)
            # NB: here we explicitly give t as input.
            x = tf.reshape(inputs[:, 0], (-1, 1))
            t = inputs[:, 1][-1]  # Now we only accept one id. We have a batch so it's a bit more complex.

            # maybe the information should come from the outside. To be defined later.

            concat = _linear([x, h_prev], 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

            dtype = inputs.dtype
            tau = vs.get_variable('tau', shape=[self._num_units],
                                  initializer=random_exp_initializer(0, self.tau_init), dtype=dtype)

            r_on = vs.get_variable('r_on', shape=[self._num_units],
                                   initializer=init_ops.constant_initializer(self.r_on_init), dtype=dtype)

            s = vs.get_variable('s', shape=[self._num_units],
                                initializer=init_ops.random_uniform_initializer(0., tau.initialized_value()),
                                dtype=dtype)

            times = tf.tile(tf.reshape(t, [-1, 1]), [1, self._num_units])
            phase = phi(times, s, tau)
            kappa = time_gate_fast(phase, r_on, self._leak_rate, self._training_phase)

            w_o_peephole = None
            if self._use_peepholes:
                w_i_peephole = vs.get_variable('W_I_peephole', shape=[self._num_units], dtype=dtype)
                w_f_peephole = vs.get_variable('W_F_peephole', shape=[self._num_units], dtype=dtype)
                w_o_peephole = vs.get_variable('W_O_peephole', shape=[self._num_units], dtype=dtype)
                f += w_f_peephole * c_prev
                i += w_i_peephole * c_prev

            new_c_tilde = sigmoid(f) * c_prev + sigmoid(i) * self._activation(j)
            if self._use_peepholes:
                o += w_o_peephole * new_c_tilde

            new_h_tilde = sigmoid(o) * self._activation(new_c_tilde)

            """
            Hi all,
            Yes, Philippe, you are correct in that Equation 4 should reference c_tilde and not c.
            I can add a point to the paper to mention that, and will update Figure 1 so the line is
            correctly drawn to c_tilde instead. The intuition here is that the gates should be blind
            to the effect of the khronos gate; input, forget and output gate should all operate as if
            the cell were a normal LSTM cell, while the khronos gate allows it to either operate or
            not operate (and then linearly interpolates between these two states). If the output gate
            is influenced by the khronos gate (if the peepholes reference c instead of c_tilde), then
            the PLSTM would no longer be a gated LSTM cell, but somehow be self-dependent on the time gate's actual operation.
            I think everyone's right in that it wouldn't influence much -- but it should be updated in
            the paper. Thanks very much for pointing out the issue, Philippe!
            -Danny"""

            # Apply Khronos gate
            new_h = kappa * new_h_tilde + (1 - kappa) * h_prev
            new_c = kappa * new_c_tilde + (1 - kappa) * c_prev
            new_state = (new_c, new_h)
            return new_h, new_state
