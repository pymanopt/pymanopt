"""
Module containing functions to differentiate functions using JAX.
"""
import functools

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as np
    from jax import jit
except ImportError:
    jax = None

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import bisect_sequence, unpack_singleton_sequence_return_value


class _JaxBackend(Backend):
    def __init__(self):
        super().__init__("Jax")

    @staticmethod
    def is_available():
        return jax is not None

    @Backend._assert_backend_available
    def is_compatible(self, objective, argument):
        return (callable(objective) and isinstance(argument, (list, tuple)) and
                len(argument) > 0)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        return jit(function)

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        num_arguments = len(arguments)
        gradient = jax.grad(function, argnums=list(range(num_arguments)))
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @staticmethod
    def _hessian_vector_product(function, argnum):
        gradient = jax.grad(function, argnum)

        def vector_dot_gradient(*args):
            arguments, vectors = args[:-1], args[-1]
            gradients = gradient(*arguments)
            return np.sum(
                [np.tensordot(gradient, vector, axes=vector.ndim)
                 for gradient, vector in zip(gradients, vectors)])
        return jax.grad(vector_dot_gradient, argnum)

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


Jax = make_tracing_backend_decorator(_JaxBackend)
