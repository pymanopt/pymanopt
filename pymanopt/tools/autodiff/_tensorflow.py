"""
Module containing functions to differentiate functions using tensorflow.
"""
try:
    import tensorflow as tf
except ImportError:
    tf = None

from warnings import warn

from ._backend import Backend, assert_backend_available


class TensorflowBackend(Backend):
    def __init__(self):
        self._session = tf.Session()

    def __str__(self):
        return "tensorflow"

    def is_available(self):
        return tf is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        if isinstance(objective, tf.Tensor):
            if (argument is None or not
                isinstance(argument, tf.Variable) and not
                all([isinstance(arg, tf.Variable)
                     for arg in argument])):
                raise ValueError(
                    "Tensorflow backend requires an argument (or sequence of "
                    "arguments) with respect to which compilation is to be "
                    "carried out")
            return True
        return False

    @assert_backend_available
    def compile_function(self, objective, argument):

        def func(x):
            feed_dict = {i: d for i, d in zip(argument, x)}
            return self._session.run(objective, feed_dict)

        return func

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Compute the gradient of 'objective' and return as a function.
        """
        tfgrad = tf.gradients(objective, argument)

        def grad(x):
            feed_dict = {i: d for i, d in zip(argument, x)}
            return self._session.run(tfgrad, feed_dict)

        return grad

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        # TODO
        raise NotImplementedError('Tensorflow backend does not yet '
                                  'implement compute_hessian.')
