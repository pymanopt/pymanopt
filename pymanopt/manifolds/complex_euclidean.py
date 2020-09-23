import numpy as np
from numpy import linalg as la, random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class _ComplexEuclidean(EuclideanEmbeddedSubmanifold):
    """Shared base class for subspace manifolds of Euclidean space."""

    def __init__(self, name, dimension, *shape):
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim/2)

    def inner(self, X, G, H):
        return np.real(np.tensordot(G.conj(), H, axes=G.ndim))

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
        return rnd.randn(*self._shape) + 1j*rnd.randn(*self._shape)

    def randvec(self, X):
        Y = self.rand()
        return Y / self.norm(X, Y)

    def transp(self, X1, X2, G):
        return G

    def pairmean(self, X, Y):
        return (X + Y) / 2

    def zerovec(self, X):
        return np.zeros(self._shape, dtype=np.complex)


class ComplexEuclidean(_ComplexEuclidean):
    """
    Complex Euclidean manifold of shape n1 x n2 x ... x nk tensors. Useful for
    unconstrained optimization problems or for unconstrained hyperparameters,
    as part of a product manifold.

    Examples:
    Create a manifold of vectors of length n:
    manifold = ComplexEuclidean(n)

    Create a manifold of m x n matrices:
    manifold = ComplexEuclidean(m, n)
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
        dimension = 2*np.prod(shape)
        super().__init__(name, dimension, *shape)
