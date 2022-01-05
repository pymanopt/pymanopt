import numpy as np
from numpy import linalg as la
from numpy import random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multiskew, multisym


class _Euclidean(EuclideanEmbeddedSubmanifold):
    """Shared base class for subspace manifolds of Euclidean space."""

    def __init__(self, name, dimension, *shape):
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, X, G, H):
        return float(np.tensordot(G, H, axes=G.ndim))

    def norm(self, X, G):
        return la.norm(G)

    def dist(self, X, Y):
        return la.norm(X - Y)

    def proj(self, X, U):
        return U

    def ehess2rhess(self, X, egrad, ehess, H):
        return ehess

    def exp(self, X, U):
        return X + U

    retr = exp

    def log(self, X, Y):
        return Y - X

    def rand(self):
        return rnd.randn(*self._shape)

    def randvec(self, X):
        Y = self.rand()
        return Y / self.norm(X, Y)

    def transp(self, X1, X2, G):
        return G

    def pairmean(self, X, Y):
        return (X + Y) / 2

    def zerovec(self, X):
        return np.zeros(self._shape)


class Euclidean(_Euclidean):
    """Euclidean manifold.

    Euclidean manifold of shape ``(n1, n2, ..., nk)`` arrays.
    Useful for unconstrained optimization problems or for unconstrained
    hyperparameters as part of a product manifold.
    """

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters")
        if len(shape) == 1:
            (n1,) = shape
            name = f"Euclidean manifold of {n1}-vectors"
        elif len(shape) == 2:
            n1, n2 = shape
            name = f"Euclidean manifold of {n1}x{n2} matrices"
        else:
            name = f"Euclidean manifold of shape {shape} tensors"
        dimension = np.prod(shape)
        super().__init__(name, dimension, *shape)


class Symmetric(_Euclidean):
    """Manifold of symmetric matrices.

    Manifold of ``n x n`` symmetric matrices as a Riemannian submanifold of
    Euclidean space.
    If ``k > 1`` then this is the product manifold of ``k`` symmetric ``n x n``
    matrices represented as arrays of shape ``(k, n, n)``.
    """

    def __init__(self, n, k=1):
        if k == 1:
            shape = (n, n)
            name = f"Manifold of {n}x{n} symmetric matrices"
        elif k > 1:
            shape = (k, n, n)
            name = f"Product manifold of {k} {n}x{n} symmetric matrices"
        else:
            raise ValueError(f"k must be an integer no less than 1, got {k}")
        dimension = int(k * n * (n + 1) / 2)
        super().__init__(name, dimension, *shape)

    def proj(self, X, U):
        return multisym(U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return multisym(ehess)

    def rand(self):
        return multisym(rnd.randn(*self._shape))

    def randvec(self, X):
        Y = self.rand()
        return multisym(Y / self.norm(X, Y))


class SkewSymmetric(_Euclidean):
    """The Euclidean space of n-by-n skew-symmetric matrices.

    If k > 1 then this is an array of shape (k, n, n) (product manifold)
    containing k (n x n) matrices.
    """

    def __init__(self, n, k=1):
        if k == 1:
            shape = (n, n)
            name = f"Manifold of {n}x{n} skew-symmetric matrices"
        elif k > 1:
            shape = (k, n, n)
            name = f"Product manifold of {k} {n}x{n} skew-symmetric matrices"
        else:
            raise ValueError("k must be an integer no less than 1")
        dimension = int(k * n * (n - 1) / 2)
        super().__init__(name, dimension, *shape)

    def proj(self, X, U):
        return multiskew(U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return multiskew(ehess)

    def rand(self):
        return multiskew(rnd.randn(*self._shape))

    def randvec(self, X):
        G = self.rand()
        return multiskew(G / self.norm(X, G))
