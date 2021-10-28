"""
Module containing functions to differentiate functions using tensorflow.
"""
try:
    import tensorflow as tf
except ImportError:  # pragma nocover
    tf = None

import functools

import numpy as np

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import bisect_sequence, unpack_singleton_sequence_return_value


class _TensorFlowBackend(Backend):
    def __init__(self, **kwargs):

        if self.is_available():
            super().__init__("TensorFlow")

    @staticmethod
    def is_available():
        return tf is not None

    @staticmethod
    def _from_numpy(array):
        """Wrap numpy ndarray ``array`` in a tensorflow tensor.
        """
        return tf.constant(array)

    def _sanitize_gradient(self, tensor, grad):
        if grad is None:
            return np.zeros_like(tensor.numpy())
        return grad.numpy()

    def _sanitize_gradients(self, tensors, grads):
        return list(map(self._sanitize_gradient, tensors, grads))

    @Backend._assert_backend_available
    def is_compatible(self, function, arguments):
        return callable(function)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        @functools.wraps(function)
        def wrapper(*args):
            return function(*map(self._from_numpy, args)).numpy()
        return wrapper

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):

        def gradient(*args):
            tf_arguments = []
            with tf.GradientTape() as tape:
                for argument in args:
                    tf_argument = self._from_numpy(argument)
                    tape.watch(tf_argument)
                    tf_arguments.append(tf_argument)
                val = function(*tf_arguments)
                grads = tape.gradient(val, tf_arguments)
            return self._sanitize_gradients(tf_arguments, grads)
        if len(arguments) == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def compute_hessian_vector_product(self, function, variables):

        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(args)
            tf_args = [self._from_numpy(arg) for arg in arguments]
            tf_vecs = [self._from_numpy(vec) for vec in vectors]
            with tf.autodiff.ForwardAccumulator(tf_args, tf_vecs) as acc:
                with tf.GradientTape() as tape:
                    for arg in tf_args:
                        tape.watch(arg)
                    val = function(*tf_args)
                    grads = tape.gradient(val, tf_args)

            return self._sanitize_gradients(tf_args, acc.jvp(grads))

        if len(variables) == 1:
            return unpack_singleton_sequence_return_value(
                    hessian_vector_product)
        return hessian_vector_product


TensorFlow = make_tracing_backend_decorator(_TensorFlowBackend)
