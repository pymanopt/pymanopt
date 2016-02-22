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
        if isinstance(argument, list):
            gradients = [grad(objective, argnum=k)
                         for k in range(0, len(argument))]

            def gradient(*args): return [gradients[k](*args)
                                         for k in range(0, len(argument))]
            return gradient
        return grad(objective)

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        if isinstance(argument, list):
            raise NotImplementedError("Computing the hessian on a "
                                      "product manifold is not yet "
                                      "implemented for the autograd "
                                      "backend ")
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
        return np.tensordot(fun_grad(*args, **kwargs), vector,
                            axes=vector.ndim)
    # Grad wrt original input.
    return grad(vector_dot_grad, argnum)
