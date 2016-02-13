"""
Module containing functions to differentiate functions using autograd.
"""
import autograd.numpy as np
import autograd as ad

from warnings import warn


def compile(objective, argument):
    """
    No compilation. Return function.
    """
    return objective


def gradient(objective, argument):
    """
    Compute the gradient of 'objective' with respect to the first argument and return
    as a function.
    """
    return ad.grad(objective)


def hessian(objective, argument):
    #TODO: cross-check, also have a look at autograd's hessian_vector_product
    hess = lambda x, g: np.tensordot(ad.hessian(objective)(x),g,axes=x.ndim)
    return hess