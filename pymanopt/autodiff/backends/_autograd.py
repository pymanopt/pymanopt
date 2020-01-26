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
from ...tools import unpack_arguments, flatten_args


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

    def _normalize_input(self, f, num_args):
        if num_args > 1:
            return unpack_arguments(f)

        @functools.wraps(f)
        def inner(x):
            # Sometimes x will be some custom type, e.g. with the
            # FixedRankEmbedded manifold. Therefore we need to cast it to a
            # numpy.ndarray.
            if isinstance(x, (list, tuple)):
                return f(list(x))
            return f(np.array(x))
        return inner

    @Backend._assert_backend_available
    def compile_function(self, func, argument):
        args = flatten_args(argument)
        if len(args) == 1:
            return func
        return unpack_arguments(func)

    @Backend._assert_backend_available
    def compute_gradient(self, func, argument):
        """
        Compute the gradient of 'func' with respect to the first
        argument and return as a function.
        """
        args = flatten_args(argument)
        if len(args) == 1:
            return autograd.grad(func)

        # Turn `func' into a function accepting a single argument which gets
        # unpacked and passed to `func' when called. This is necessary so we
        # can compute the gradient for multiple arguments at once by computing
        # the gradient of the wrapper function instead. In order to unpack
        # arguments appropriately, we need a signature hint in the form of
        # `argument'. This is because autograd wraps tuples and lists in a
        # `SequenceBox' which is not a subclass of tuple or list so we cannot
        # detect nested tuples as call arguments without a signature hint.
        unary_func = unpack_arguments(func, signature=argument)
        grad = autograd.grad(unary_func)
        return grad

    @staticmethod
    def _compute_unary_hvp(func):
        """
        Builds a function that returns the exact Hessian-vector product for a
        regular unary function.
        """
        grad = autograd.grad(func)

        def vector_dot_grad(*args):
            args, vector = args[:-1], args[-1]
            return np.tensordot(grad(*args), vector, axes=vector.ndim)
        return autograd.grad(vector_dot_grad)

    @staticmethod
    def _compute_nary_hvp(func, argument):
        """
        Builds a function that returns the exact Hessian-vector product for a
        function accepting one or more (possibly nested) arguments.
        """
        grad = autograd.grad(func)

        def vector_dot_grad(*args):
            args, vectors = args[:-1], args[-1]
            g = grad(*args)
            return np.sum(
                [np.tensordot(g[k], vector, axes=vector.ndim)
                 for k, vector in enumerate(vectors)])
        return autograd.grad(vector_dot_grad)

    @Backend._assert_backend_available
    def compute_hessian(self, func, argument):
        args = flatten_args(argument)
        if len(args) == 1:
            return self._compute_unary_hvp(func)
        return self._compute_nary_hvp(func, argument)


Autograd = make_tracing_backend_decorator(_AutogradBackend)
