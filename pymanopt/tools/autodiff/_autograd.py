"""
Module containing functions to differentiate functions using autograd.
"""
import autograd.numpy as np
import autograd

from ._backend import Backend


class AutogradBackend(Backend):
    def compute_gradient(self, objective, argument):
        """
        Compute the gradient of 'objective' with respect to the first
        argument and return as a function.
        """
        return autograd.grad(objective)

    def compute_hessian(self, objective, argument):
        # TODO: cross-check, also have a look at autograd's
        #       hessian_vector_product
        def hess(x, g):
            return np.tensordot(autograd.hessian(objective)(x), g, axes=x.ndim)
        return hess
