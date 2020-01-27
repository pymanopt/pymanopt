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
from ...tools import flatten_arguments, group_return_values


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
        flattened_arguments = flatten_arguments(arguments)
        return all([isinstance(argument, tf.Variable)
                    for argument in flattened_arguments])

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)
        if len(flattened_arguments) == 1:
            def unary_function(point):
                (argument,) = flattened_arguments
                feed_dict = {argument: point}
                return self._session.run(function, feed_dict)
            return unary_function

        def nary_function(arguments):
            flattened_inputs = flatten_arguments(arguments)
            feed_dict = {
                argument: array
                for argument, array in zip(flattened_arguments,
                                           flattened_inputs)
            }
            return self._session.run(function, feed_dict)
        return nary_function

    @staticmethod
    def _gradients(function, arguments):
        return tf.gradients(function, arguments,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)
        gradient = self._gradients(function, flattened_arguments)

        if len(flattened_arguments) == 1:
            (argument,) = flattened_arguments

            def unary_gradient(point):
                feed_dict = {argument: point}
                return self._session.run(gradient[0], feed_dict)
            return unary_gradient

        def nary_gradient(points):
            feed_dict = {
                argument: point
                for argument, point in zip(flattened_arguments,
                                           flatten_arguments(points))
            }
            return self._session.run(gradient, feed_dict)
        return group_return_values(nary_gradient, arguments)

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
    def compute_hessian(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)

        if len(flattened_arguments) == 1:
            (argument,) = flattened_arguments
            zeros = tf.zeros_like(argument)
            hessian = self._hessian_vector_product(
                function, [argument], [zeros])

            def unary_hessian(point, vector):
                feed_dict = {argument: point, zeros: vector}
                return self._session.run(hessian[0], feed_dict)
            return unary_hessian

        zeros = [tf.zeros_like(argument) for argument in flattened_arguments]
        hessian = self._hessian_vector_product(
            function, flattened_arguments, zeros)

        def nary_hessian(points, vectors):
            feed_dict = {
                argument: value for argument, value in zip(
                    itertools.chain(flattened_arguments, zeros),
                    itertools.chain(flatten_arguments(points),
                                    flatten_arguments(vectors)))
            }
            return self._session.run(hessian, feed_dict)
        return group_return_values(nary_hessian, arguments)


TensorFlow = make_graph_backend_decorator(_TensorFlowBackend)
