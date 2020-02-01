"""
Module containing functions to differentiate functions using autograd.
"""
import functools

try:
    import autograd
except ImportError:
    autograd = None

from ._backend import Backend
from .. import make_tracing_backend_decorator


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

    def _unpack_return_value(self, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)[0]
        return wrapper

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        num_arguments = len(arguments)
        gradient = autograd.grad(function, argnum=list(range(num_arguments)))
        if num_arguments > 1:
            return gradient
        return self._unpack_return_value(gradient)

    @Backend._assert_backend_available
    def compute_hessian_vector_product(self, function, arguments):
        num_arguments = len(arguments)
        hessian_vector_product = autograd.hessian_vector_product(
            function, argnum=tuple(range(num_arguments)))
        if num_arguments == 1:
            return self._unpack_return_value(hessian_vector_product)

        @functools.wraps(hessian_vector_product)
        def wrapper(*arguments):
            num_arguments = len(arguments)
            assert num_arguments % 2 == 0
            point = arguments[:num_arguments // 2]
            vector = arguments[num_arguments // 2:]
            return hessian_vector_product(*point, vector)
        return wrapper


Autograd = make_tracing_backend_decorator(_AutogradBackend)
