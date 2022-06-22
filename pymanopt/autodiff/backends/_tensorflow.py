try:
    import tensorflow as tf
except ImportError:  # pragma nocover
    tf = None

import functools

import numpy as np

from ...tools import bisect_sequence, unpack_singleton_sequence_return_value
from ._backend import Backend


class TensorFlowBackend(Backend):
    def __init__(self):
        super().__init__("TensorFlow")

    @staticmethod
    def is_available():
        return tf is not None

    @staticmethod
    def _from_numpy(array):
        """Wrap numpy ndarray ``array`` in a tensorflow tensor."""
        return tf.constant(array)

    def _sanitize_gradient(self, tensor, grad):
        if grad is None:
            return np.zeros_like(tensor.numpy())
        return grad.numpy()

    def _sanitize_gradients(self, tensors, grads):
        return list(map(self._sanitize_gradient, tensors, grads))

    @Backend._assert_backend_available
    def prepare_function(self, function):
        @functools.wraps(function)
        def wrapper(*args):
            return function(*map(self._from_numpy, args)).numpy()

        return wrapper

    @Backend._assert_backend_available
    def generate_gradient_operator(self, function, num_arguments):
        def gradient(*args):
            arguments = list(map(self._from_numpy, args))
            with tf.GradientTape() as tape:
                for argument in arguments:
                    tape.watch(argument)
                gradients = tape.gradient(function(*arguments), arguments)
            return self._sanitize_gradients(arguments, gradients)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def generate_hessian_operator(self, function, num_arguments):
        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(
                list(map(self._from_numpy, args))
            )
            with tf.GradientTape() as tape, tf.autodiff.ForwardAccumulator(
                arguments, vectors
            ) as accumulator:
                for argument in arguments:
                    tape.watch(argument)
                gradients = tape.gradient(function(*arguments), arguments)
            return self._sanitize_gradients(
                arguments, accumulator.jvp(gradients)
            )

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(
                hessian_vector_product
            )
        return hessian_vector_product
