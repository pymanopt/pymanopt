"""
Module containing functions to differentiate functions using autograd.
"""
import functools

try:
    import autograd
    import autograd.numpy as np
except ImportError:
    autograd = None

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import bisect_sequence, unpack_singleton_sequence_return_value


class _AutogradBackend(Backend):
    def __init__(self):
        super().__init__("Autograd")

    @staticmethod
    def is_available():
        return autograd is not None

    @Backend._assert_backend_available
    def is_compatible(self, objective, argument):
        return (callable(objective) and isinstance(argument, (list, tuple)) and
                len(argument) > 0)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        return function

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        num_arguments = len(arguments)
        gradient = autograd.grad(function, argnum=list(range(num_arguments)))
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @staticmethod
    def _hessian_vector_product(function, argnum):
        gradient = autograd.grad(function, argnum)

        def vector_dot_gradient(*args):
            arguments, vectors = args[:-1], args[-1]
            gradients = gradient(*arguments)
            return np.sum(
                [np.tensordot(gradient, vector, axes=vector.ndim)
                 for gradient, vector in zip(gradients, vectors)])
        return autograd.grad(vector_dot_gradient, argnum)

    @Backend._assert_backend_available
    def compute_hessian_vector_product(self, function, arguments):
        num_arguments = len(arguments)
        hessian_vector_product = self._hessian_vector_product(
            function, argnum=tuple(range(num_arguments)))

        @functools.wraps(hessian_vector_product)
        def wrapper(*args):
            arguments, vectors = bisect_sequence(args)
            return hessian_vector_product(*arguments, vectors)
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper


Autograd = make_tracing_backend_decorator(_AutogradBackend)
