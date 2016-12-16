from tensorflow.python.ops import array_ops, math_ops, init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.util import nest


def phi(t, s, tau):
    return ((t - s) % tau) / tau


def time_gate(phase, r_on, leak_rate):
    if phase < 0.5 * r_on:
        return 2 * phase / r_on
    elif 0.5 * r_on < phase < r_on:
        2 - 2 * phase / r_on
    else:
        leak_rate * phase


# this is going to change with v0.13

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    '''Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to 'Linear'.

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    '''
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError('`args` must be specified')
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))
        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or 'Linear'):
        matrix = vs.get_variable(
            'Matrix', [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            'Bias', [output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


class PhasedLSTMCell(RNNCell):
    def __init__(self, num_units, use_peepholes=True, activation=tanh):
        self._num_units = num_units
        self._activation = activation
        self._use_peepholes = use_peepholes

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """ Long short-term phased memory cell (P-LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):
            # Parameters of gates are concatenated into one multiply for efficiency.
            c_prev, h_prev = state
            concat = _linear([inputs, h_prev], 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(1, 4, concat)

            if self._use_peepholes:
                w_i_peephole = vs.get_variable('W_I_peephole', shape=[self._num_units], dtype=inputs.dtype)
                w_f_peephole = vs.get_variable('W_F_peephole', shape=[self._num_units], dtype=inputs.dtype)
                w_o_peephole = vs.get_variable('W_O_peephole', shape=[self._num_units], dtype=inputs.dtype)

                new_c = (sigmoid(f + w_f_peephole * c_prev) * c_prev +
                         sigmoid(i + w_i_peephole * c_prev) * self._activation(j))
                new_h = sigmoid(o + w_o_peephole * new_c) * self._activation(new_c)
            else:
                new_c = (sigmoid(f) * c_prev + sigmoid(i) * self._activation(j))
                new_h = sigmoid(o) * self._activation(new_c)

            new_state = (new_c, new_h)
            return new_h, new_state
