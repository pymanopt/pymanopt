"""Tools for testing numerical correctness in Pymanopt.

Notes:
    The functions :func:`rgrad`, :func:`euclidean_to_riemannian_gradient`,
    :func:`ehess` and :func:`ehess2rhess` will only be correct if the manifold
    is a submanifold of Euclidean space, that is if the projection is an
    orthogonal projection onto the tangent space.
"""

import numpy as np
from autograd import grad, jacobian


def riemannian_gradient(cost, projector):
    """Generates the Riemannian gradient of a cost function."""

    def gradient_function(point):
        return projector(point, grad(cost)(point))

    return gradient_function


def euclidean_to_riemannian_gradient(projector):
    """Generates an euclidean_to_riemannian_gradient function."""

    def converter(point, euclidean_gradient):
        return projector(point, euclidean_gradient)

    return converter


def ehess2rhess(projector):
    """Generates an ehess2rhess function.

    Specifically, ``ehess2rhess(proj)(point, euclidean_gradient, euclidean_hvp,
    tangent_vector)`` converts the Euclidean Hessian-vector product
    ``euclidean_hvp`` at a point ``point`` to a Riemannian Hessian-vector
    product, i.e., the directional derivative of the gradient in the tangent
    direction ``tangent_vector``.
    Similar to :func:`rhess`, this is not efficient as it computes the Jacobian
    explicitly.
    """
    jacobian_projector = jacobian(projector)

    def converter(point, euclidean_gradient, euclidean_hvp, tangent_vector):
        return projector(
            point,
            euclidean_hvp
            + np.tensordot(
                jacobian_projector(point, euclidean_gradient),
                tangent_vector,
                axes=tangent_vector.ndim,
            ),
        )

    return converter
