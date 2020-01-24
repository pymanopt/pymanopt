"""
Module containing functions to differentiate functions using autograd.
"""
import functools

try:
    import autograd
    from autograd import numpy as np
except ImportError:
    autograd = None

from ._backend import Backend, assert_backend_available
from .. import make_tracing_backend_decorator
from ...tools import unpack_arguments, flatten_args, group_return_values


class AutogradBackend(Backend):
    def __str__(self):
        return "autograd"

    @staticmethod
    def is_available():
        return autograd is not None

    @assert_backend_available
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

    @assert_backend_available
    def compile_function(self, objective, argument):
        return self._normalize_input(objective, len(argument))

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Compute the gradient of 'objective' with respect to the first
        argument and return as a function.
        """
        assert autograd is not None
        g = autograd.grad(objective)
        return self._normalize_input(g, len(argument))

    @staticmethod
    def _hessian_vector_product(fun):
        """Builds a function that returns the exact Hessian-vector product.
        The returned function has arguments (*args, vector, **kwargs). Note,
        this function will be incorporated into autograd, with name
        hessian_vector_product. Once it has been this function can be
        deleted."""
        assert autograd is not None
        fun_grad = autograd.grad(fun)

        def vector_dot_grad(*args, **kwargs):
            args, vector = args[:-1], args[-1]
            # Assume we are on the product manifold.
            return np.sum([np.tensordot(fun_grad(*args, **kwargs)[k],
                                        vector[k], axes=vector[k].ndim)
                           for k in range(len(vector))])
        # Grad wrt original input.
        return autograd.grad(vector_dot_grad)

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        try:
            return autograd.hessian_vector_product(objective)
        except AttributeError:
            pass
        return self._hessian_vector_product(objective)


Autograd = make_tracing_backend_decorator(AutogradBackend)
