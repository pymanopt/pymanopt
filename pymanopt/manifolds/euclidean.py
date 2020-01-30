import numpy as np
from numpy import linalg as la, random as rnd

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
    """
    Euclidean manifold of shape n1 x n2 x ... x nk tensors. Useful for
    unconstrained optimization problems or for unconstrained hyperparameters,
    as part of a product manifold.

    Examples:
    Create a manifold of vectors of length n:
    manifold = Euclidean(n)

    Create a manifold of m x n matrices:
    manifold = Euclidean(m, n)
    """

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters")
        if len(shape) == 1:
            name = "Euclidean manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            name = ("Euclidean manifold of {}x{} matrices").format(*shape)
        else:
            name = ("Euclidean manifold of shape " + str(shape) + " tensors")
        dimension = np.prod(shape)
        super().__init__(name, dimension, *shape)


class Symmetric(_Euclidean):
    """
    Manifold of n x n symmetric matrices, as a Riemannian submanifold of
    Euclidean space.

    If k > 1 then this is an array of shape (k, n, n) (product manifold)
    containing k (n x n) matrices.
    """

    def __init__(self, n, k=1):
        if k == 1:
            shape = (n, n)
            name = ("Manifold of {} x {} symmetric matrices").format(n, n)
        elif k > 1:
            shape = (k, n, n)
            name = ("Product manifold of {} ({} x {}) symmetric "
                    "matrices").format(k, n, n)
        else:
            raise ValueError("k must be an integer no less than 1")
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
    """
    The Euclidean space of n-by-n skew-symmetric matrices.

    If k > 1 then this is an array of shape (k, n, n) (product manifold)
    containing k (n x n) matrices.
    """

    def __init__(self, n, k=1):
        if k == 1:
            shape = (n, n)
            name = ("Manifold of {} x {} skew-symmetric "
                    "matrices").format(n, n)
        elif k > 1:
            shape = (k, n, n)
            name = ("Product manifold of {} ({} x {}) skew-symmetric "
                    "matrices").format(k, n, n)
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
