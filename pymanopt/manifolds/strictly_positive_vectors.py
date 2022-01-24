import numpy as np
from numpy import linalg as la
from numpy import random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class StrictlyPositiveVectors(EuclideanEmbeddedSubmanifold):
    r"""Manifold of strictly positive vectors.

    Since :math:`((\R_{++})^n)^k` is isomorphic to the manifold of positive
    definite diagonal matrices the geometry is inherited from the geometry of
    positive definite matrices.
    """

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of strictly positive {n}-vectors"
        else:
            name = f"Product manifold of {k} strictly positive {n}-vectors"
        dimension = int(k * n)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, x, u, v):
        inv_x = 1.0 / x
        return np.sum(inv_x * u * inv_x * v)

    def proj(self, x, u):
        return u

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def rand(self):
        k, n = self._k, self._n
        if k == 1:
            x = rnd.uniform(low=1e-6, high=1, size=(n, 1))
        else:
            x = rnd.uniform(low=1e-6, high=1, size=(k, n, 1))
        return x

    def randvec(self, x):
        k, n = self._k, self._n
        if k == 1:
            u = rnd.normal(size=(n, 1))
        else:
            u = rnd.normal(size=(k, n, 1))
        return u / self.norm(x, u)

    def zerovec(self, x):
        k, n = self._k, self._n
        if k == 1:
            u = np.zeros((n, 1))
        else:
            u = np.zeros((k, n, 1))
        return u

    def dist(self, x, y):
        return la.norm(np.log(x) - np.log(y))

    def egrad2rgrad(self, x, u):
        return u * (x ** 2)

    def ehess2rhess(self, x, egrad, ehess, u):
        return ehess * (x ** 2) + egrad * u * x

    def exp(self, x, u):
        return x * np.exp((1.0 / x) * u)

    # order 2 retraction
    def retr(self, x, u):
        return x + u + (1 / 2) * (x ** -1) * (u ** 2)

    def log(self, x, y):
        return x * np.log((1.0 / x) * y)

    def transp(self, x1, x2, d):
        return self.proj(x2, x2 * (x1 ** -1) * d)
