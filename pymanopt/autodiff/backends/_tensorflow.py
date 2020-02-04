"""
Module containing functions to differentiate functions using tensorflow.
"""
import itertools

try:
    import tensorflow as tf
except ImportError:
    tf = None

from ._backend import Backend
from .. import make_graph_backend_decorator
from ...tools import bisect_sequence, unpack_singleton_sequence_return_value


class _TensorFlowBackend(Backend):
    def __init__(self, **kwargs):
        self._own_session = None

        if self.is_available():
            self._session = kwargs.get("session")
            if self._session is None:
                self._own_session = self._session = tf.Session()
        super().__init__("TensorFlow")

    def __del__(self):
        if self._own_session is not None:
            self._own_session.close()
            self._session = self._own_session = None

    @staticmethod
    def is_available():
        return tf is not None

    @Backend._assert_backend_available
    def is_compatible(self, function, arguments):
        if not isinstance(function, tf.Tensor):
            return False
        return all([isinstance(argument, tf.Variable)
                    for argument in arguments])

    @Backend._assert_backend_available
    def compile_function(self, function, variables):
        def compiled_function(*args):
            feed_dict = {
                variable: argument
                for variable, argument in zip(variables, args)
            }
            return self._session.run(function, feed_dict)
        return compiled_function

    @staticmethod
    def _gradients(function, arguments):
        return tf.gradients(function, arguments,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @Backend._assert_backend_available
    def compute_gradient(self, function, variables):
        gradients = self._gradients(function, variables)

        def gradient(*args):
            feed_dict = {
                variable: argument
                for variable, argument in zip(variables, args)
            }
            return self._session.run(gradients, feed_dict)
        if len(variables) == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @staticmethod
    def _hessian_vector_product(function, arguments, vectors):
        """Multiply the Hessian of `function` w.r.t. `arguments` by `vectors`.

        Notes
        -----
        The implementation of this method is based on TensorFlow's
        '_hessian_vector_product' [1]_. The (private) '_hessian_vector_product'
        TensorFlow function replaces unconnected gradients with None, which
        results in exceptions when a function depends linearly on one of its
        inputs. Instead, we here allow unconnected gradients to be zero.

        References
        ----------
        [1] https://git.io/JvmrW
        """

        # Validate the input
        num_arguments = len(arguments)
        assert len(vectors) == num_arguments

        # First backprop
        gradients = _TensorFlowBackend._gradients(function, arguments)

        assert len(gradients) == num_arguments
        element_wise_products = [
            tf.multiply(gradient, tf.stop_gradient(vector))
            for gradient, vector in zip(gradients, vectors)
            if gradient is not None
        ]

        # Second backprop
        return _TensorFlowBackend._gradients(element_wise_products, arguments)

    @Backend._assert_backend_available
    def compute_hessian_vector_product(self, function, variables):
        zeros = [tf.zeros_like(variable) for variable in variables]
        hessian = self._hessian_vector_product(function, variables, zeros)

        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(args)
            feed_dict = {
                variable: argument
                for variable, argument in zip(
                    itertools.chain(variables, zeros),
                    itertools.chain(arguments, vectors))
            }
            return self._session.run(hessian, feed_dict)
        if len(variables) == 1:
            return unpack_singleton_sequence_return_value(
                hessian_vector_product)
        return hessian_vector_product


TensorFlow = make_graph_backend_decorator(_TensorFlowBackend)
