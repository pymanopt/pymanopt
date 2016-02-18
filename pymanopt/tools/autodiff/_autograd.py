"""
Module containing functions to differentiate functions using autograd.
"""
import autograd.numpy as np
from autograd.core import grad

from ._backend import Backend


def _hessian_vector_product(fun, argnum=0):
    """Builds a function that returns the exact Hessian-vector product.
    The returned function has arguments (*args, vector, **kwargs). Note,
    this function will be incorporated into autograd, with name
    hessian_vector_product. Once it has been this function can be
    deleted."""
    fun_grad = grad(fun, argnum)

    def vector_dot_grad(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.tensordot(fun_grad(*args, **kwargs), vector,
                            axes=vector.ndim)
    # Grad wrt original input.
    return grad(vector_dot_grad, argnum)


class AutogradBackend(Backend):
    def compute_gradient(self, objective, argument):
        """
        Compute the gradient of 'objective' with respect to the first
        argument and return as a function.
        """
        return grad(objective)

    def compute_hessian(self, objective, argument):
        return _hessian_vector_product(objective)(x, g)
