"""
Module containing functions to differentiate functions using autograd.
"""
import functools

try:
    import autograd
    from autograd import numpy as np
except ImportError:
    autograd = None

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import flatten_arguments, group_return_values, unpack_arguments


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
        flattened_arguments = flatten_arguments(arguments)
        if len(flattened_arguments) == 1:
            return function
        return unpack_arguments(function)

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)
        if len(flattened_arguments) == 1:
            return autograd.grad(function)
        # XXX: This path handles cases where the signature hint looks like
        #      '@Autograd(("x", "y"))'. This is potentially unnecessary as
        #      tests also pass if we instead use '@Autograd'. Revisit this
        #      once we ported more complicated examples to autograd.
        if len(arguments) == 1:
            @functools.wraps(function)
            def unary_function(arguments):
                return function(*arguments)
            return autograd.grad(unary_function)

        # Turn `function` into a function accepting a single argument which
        # gets unpacked when the function is called. This is necessary for
        # autograd to compute and return the gradient for each input in the
        # input tuple/list and return it in the same grouping.
        # In order to unpack arguments correctly, we also need a signature hint
        # in the form of `arguments`. This is because autograd wraps tuples and
        # lists in `SequenceBox` types which are not derived from tuple or list
        # so we cannot detect nested arguments automatically.
        unary_function = unpack_arguments(function, signature=arguments)
        return autograd.grad(unary_function)

    @staticmethod
    def _compute_nary_hessian_vector_product(function):
        gradient = autograd.grad(function)

        def vector_dot_grad(*args):
            arguments, vectors = args[:-1], args[-1]
            gradients = gradient(*arguments)
            return np.sum(
                [np.tensordot(gradient, vector, axes=vector.ndim)
                 for gradient, vector in zip(gradients, vectors)])
        return autograd.grad(vector_dot_grad)

    @Backend._assert_backend_available
    def compute_hessian(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)
        if len(flattened_arguments) == 1:
            return autograd.hessian_tensor_product(function)
        if len(arguments) == 1:
            @functools.wraps(function)
            def unary_function(arguments):
                return function(*arguments)
            return autograd.hessian_tensor_product(unary_function)

        @functools.wraps(function)
        def unary_function(arguments):
            return function(*arguments)
        hessian_vector_product = self._compute_nary_hessian_vector_product(
            unary_function)

        @functools.wraps(hessian_vector_product)
        def wrapper(point, vector):
            return hessian_vector_product(
                flatten_arguments(point, signature=arguments),
                flatten_arguments(vector, signature=arguments))
        return group_return_values(wrapper, arguments)


Autograd = make_tracing_backend_decorator(_AutogradBackend)
