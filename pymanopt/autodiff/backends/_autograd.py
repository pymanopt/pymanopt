import functools


try:
    import autograd
    import autograd.numpy as np
except ImportError:
    autograd = None

from ...tools import bisect_sequence, unpack_singleton_sequence_return_value
from ._backend import Backend


class AutogradBackend(Backend):
    def __init__(self):
        super().__init__("Autograd")

    @staticmethod
    def is_available():
        return autograd is not None

    @Backend._assert_backend_available
    def prepare_function(self, function):
        return function

    @Backend._assert_backend_available
    def generate_gradient_operator(self, function, num_arguments):
        gradient = autograd.grad(function, argnum=list(range(num_arguments)))
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @staticmethod
    def _hessian_vector_product(function, argnum):
        gradient = autograd.grad(function, argnum)

        def vector_dot_gradient(*args):
            *arguments, vectors = args
            gradients = gradient(*arguments)
            return np.sum(
                [
                    np.tensordot(gradient, vector, axes=vector.ndim)
                    for gradient, vector in zip(gradients, vectors)
                ]
            )

        return autograd.grad(vector_dot_gradient, argnum)

    @Backend._assert_backend_available
    def generate_hessian_operator(self, function, num_arguments):
        hessian_vector_product = self._hessian_vector_product(
            function, argnum=list(range(num_arguments))
        )

        @functools.wraps(hessian_vector_product)
        def wrapper(*args):
            arguments, vectors = bisect_sequence(args)
            return hessian_vector_product(*arguments, vectors)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper
