import numpy as np
from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multiskew


class Skew_Symmetric(Manifold):
    """
    The Euclidean space of n-by-n skew-symmetric matrices.

    If k > 1 then this is an array of shape (k, n, n) (product manifold)
    containing k (n x n) matrices.
    """

    def __init__(self, n, k=1):
        if (k == 1):
            self._shape = (n, n)
            self._name = ("Manifold of {} x {} skew-symmetric matrices."
                          ).format(n, n)
        elif(k > 1):
            self._shape = (k, n, n)
            self._name = ("Product manifold of {} ({} x {}) skew-symmetric "
                          "matrices.").format(k, n, n)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._n = n
        self._k = k
        self._dim = 0.5 * self._k * self._n * (self._n - 1)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    def norm(self, X, G):
        return np.linalg.norm(G)

    def inner(self, X, G, H):
        return float(np.tensordot(G, H, axes=G.ndim))

    def dist(self, X, Y):
        return np.linalg.norm(X - Y)

    @property
    def typicaldist(self):
        return np.sqrt(self._k) * self._n

    def proj(self, X, U):
        return multiskew(U)

    def egrad2rgrad(self, X, U):
        return self.proj(X, U)

    def tangent(self, X, U):
        return self.proj(X, U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return self.proj(X, ehess)

    def rand(self):
        return multiskew(np.random.randn(*self._shape))

    def randvec(self, X):
        G = self.rand()
        return multiskew(G / self.norm(X, G))

    def exp(self, X, G):
        return X + G

    def retr(self, X, G):
        return self.exp(X, G)

    def log(self, X, Y):
        return Y - X

    def transp(self, x1, x2, d):
        return d

    def pairmean(self, X, Y):
        return 0.5 * (X + Y)
