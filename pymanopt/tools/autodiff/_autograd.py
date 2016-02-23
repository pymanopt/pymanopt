"""
Module containing functions to differentiate functions using autograd.
"""
try:
    import autograd.numpy as np
    from autograd.core import grad
except ImportError:
    np = None
    grad = None

from ._backend import Backend, assert_backend_available


class AutogradBackend(Backend):
    @property
    def name(self):
        return "autograd"

    def is_available(self):
        return np is not None and grad is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        return callable(objective)

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Compute the gradient of 'objective' with respect to the first
        argument and return as a function.
        """
        return grad(objective)

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        return _hessian_vector_product(objective)


def _hessian_vector_product(fun, argnum=0):
    """Builds a function that returns the exact Hessian-vector product.
    The returned function has arguments (*args, vector, **kwargs). Note,
    this function will be incorporated into autograd, with name
    hessian_vector_product. Once it has been this function can be
    deleted."""
    fun_grad = grad(fun, argnum)

    def vector_dot_grad(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        try:
            return np.tensordot(fun_grad(*args, **kwargs), vector,
                                axes=vector.ndim)
        except AttributeError:
            # Assume we are on the product manifold.
            return np.sum([np.tensordot(fun_grad(*args, **kwargs)[k],
                                        vector[k], axes=vector[k].ndim)
                           for k in range(len(vector))])
    # Grad wrt original input.
    return grad(vector_dot_grad, argnum)
