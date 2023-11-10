"""Tools for testing numerical correctness in Pymanopt.

Note:
    The functions :func:`rgrad`, :func:`euclidean_to_riemannian_gradient`,
    :func:`ehess` and :func:`euclidean_to_riemannian_hessian` will only be
    correct if the manifold is a submanifold of Euclidean space, that is if the
    projection is an orthogonal projection onto the tangent space.
"""

from copy import deepcopy
from torch.autograd import grad
from torch.autograd.functional import jacobian


import pymanopt.numerics as nx


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


def euclidean_to_riemannian_hessian(projector):
    """Generates an euclidean_to_riemannian_hessian function.

    Specifically, ``euclidean_to_riemannian_hessian(proj)(point,
    euclidean_gradient, euclidean_hessian, tangent_vector)`` converts the Euclidean
    Hessian-vector product ``euclidean_hessian`` at a point ``point`` to a
    Riemannian Hessian-vector product, i.e., the directional derivative of the
    gradient in the tangent direction ``tangent_vector``.
    Similar to :func:`riemannian_hessian`, this is not efficient as it computes the
    Jacobian explicitly.
    """

    def converter(
        point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        # keep a copy of point for backend compatibility
        point_copy = deepcopy(point)

        # tranform everything to numpy arrays
        point = nx.to_backend(point, 'pytorch')
        euclidean_gradient = nx.to_backend(euclidean_gradient, 'pytorch')
        euclidean_hessian = nx.to_backend(euclidean_hessian, 'pytorch')
        tangent_vector = nx.to_backend(tangent_vector, 'pytorch')

        # compute the Riemannian Hessian
        riemannian_hessian = projector(
            point,
            euclidean_hessian
            + nx.tensordot(
                jacobian(projector, (point, euclidean_gradient))[0],
                tangent_vector,
                axes=nx.ndim(tangent_vector)
            ),
        )

        # convert back to the original backend and return
        return nx.array_as(riemannian_hessian, as_=point_copy)

    return converter
