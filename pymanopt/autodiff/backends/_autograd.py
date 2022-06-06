import functools


try:
    import autograd
except ImportError:
    autograd = None

from ...tools import bisect_sequence, unpack_singleton_sequence_return_value
from ._backend import Backend, fail_on_complex_input


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
        gradient = autograd.grad(
            fail_on_complex_input(function), argnum=list(range(num_arguments))
        )
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def generate_hessian_operator(self, function, num_arguments):
        wrapped_function = fail_on_complex_input(function)
        argnum = list(range(num_arguments))

        def hessian_vector_product(*arguments, vectors):
            return autograd.make_jvp(
                autograd.grad(wrapped_function, argnum), argnum
            )(*arguments)(vectors)[1]

        @functools.wraps(hessian_vector_product)
        def wrapper(*args):
            arguments, vectors = bisect_sequence(args)
            return hessian_vector_product(*arguments, vectors=vectors)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper
