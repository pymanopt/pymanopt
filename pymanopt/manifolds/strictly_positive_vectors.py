import numpy as np
from numpy import linalg as la, random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class StrictlyPositiveVectors(EuclideanEmbeddedSubmanifold):
    """Manifold of k strictly positive n-dimensional vectors, denoted ((R++)^n)^k.
    Since ((R++)^n)^k is isomorphic to
    (D_n^{++})^k (manifold of positive definite diagonal matrices of size n),
    the geometry is inherited of the positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = ("Manifold of strictly positive vectors of size{}").format(
                n, n)
        else:
            name = ("Product manifold of {} \
                    strictly positive vectors of size {}").format(k, n)
        dimension = int(k * n)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, x, u, v):
        inv_x = (1./x)
        return np.tensordot(inv_x*u, inv_x*v, axes=(-1, -1))

    def proj(self, x, u):
        return u

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def rand(self):
        if self._k == 1:
            return rnd.uniform(low=1e-6, high=1, size=(self._n))
        return rnd.uniform(low=1e-6, high=1, size=(self._k, self._n))

    def randvec(self, x):
        if self._k == 1:
            u = rnd.randn(self._n)
        else:
            u = rnd.randn(self._k, self._n)
        return u / self.norm(x, u)

    def zerovec(self, x):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros(n)
        return np.zeros(k, n)

    def dist(self, x, y):
        return la.norm(np.log(x)-np.log(y), axis=0)

    egrad2rgrad = proj

    # def ehess2rhess(self, x, egrad, ehess, u):

    def exp(self, x, u):
        return x*np.exp((1./x)*u)

    def retr(self, x, u):
        return x+u

    def log(self, x, y):
        return x*np.log((1./x)*y)

# def transp(self, x1, x2, d):
