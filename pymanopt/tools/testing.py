"""Tools for testing numerical correctness in Pymanopt.

Note:
    The functions :func:`rgrad`, :func:`egrad2rgrad`, :func:`ehess` and
    :func:`ehess2rhess` will only be correct if the manifold is a submanifold
    of Euclidean space, that is if the projection is an orthogonal projection
    onto the tangent space.
"""

import numpy as np
from autograd import grad, jacobian


def rgrad(cost, proj):
    """Generates the Riemannian gradient of a cost function."""
    return lambda x: proj(x, grad(cost)(x))


def egrad2rgrad(proj):
    """Generates an egrad2rgrad function."""
    return lambda x, g: proj(x, g)


def rhess(cost, proj):
    """Generates the Riemannian hessian of a cost function.

    Specifically, ``rhess(cost, proj)(x, u)`` is the directional derivative of
    cost at ``x`` on the manifold in the direction of a tangent vector ``u``.
    Both ``cost`` and ``proj`` must be defined using Autograd.
    The current implementation is not efficient because of the explicit
    Jacobian-vector product.
    """
    return lambda x, u: proj(
        x, np.tensordot(jacobian(rgrad(cost, proj))(x), u, axes=u.ndim)
    )


def ehess2rhess(proj):
    """Generates an ehess2rhess function.

    Specifically, ``ehess2rhess(proj)(x, egrad, ehess, u)`` converts the
    Euclidean Hessian-vector product ``ehess`` at a point ``x`` to a Riemannian
    Hessian-vector product, i.e., the directional derivative of the gradient in
    the tangent direction ``u``.
    Similar to :func:`rhess`, this is not efficient as it computes the Jacobian
    explicitly.
    """
    # Differentiate proj w.r.t. the first argument
    d_proj = jacobian(proj)
    return lambda x, egrad, ehess, u: proj(
        x, ehess + np.tensordot(d_proj(x, egrad), u, axes=u.ndim)
    )
