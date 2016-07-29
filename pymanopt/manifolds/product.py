import numpy as np

from pymanopt.manifolds.manifold import Manifold


class Product(Manifold):
    """
    Product manifold, i.e. the cartesian product of multiple manifolds.
    """

    def __init__(self, manifolds):
        self._manifolds = manifolds
        self._nmanifolds = len(manifolds)

    def __str__(self):
        return ("Product manifold: {:s}".format(
                " X ".join([str(man) for man in self._manifolds])))

    @property
    def dim(self):
        return np.sum([man.dim for man in self._manifolds])

    @property
    def typicaldist(self):
        return np.sqrt(np.sum([man.typicaldist**2 for man in self._manifolds]))

    def inner(self, X, G, H):
        return np.sum([self._manifolds[k].inner(X[k], G[k], H[k])
                       for k in range(0, self._nmanifolds)])

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        return np.sqrt(np.sum([self._manifolds[k].dist(X[k], Y[k])**2
                               for k in range(0, self._nmanifolds)]))

    def proj(self, X, U):
        return _TangentVector([self._manifolds[k].proj(X[k], U[k])
                               for k in range(0, self._nmanifolds)])

    def egrad2rgrad(self, X, U):
        return _TangentVector([self._manifolds[k].egrad2rgrad(X[k], U[k])
                              for k in range(0, self._nmanifolds)])

    def ehess2rhess(self, X, egrad, ehess, H):
        return _TangentVector([self._manifolds[k].ehess2rhess(X[k], egrad[k],
                                                              ehess[k], H[k])
                               for k in range(0, self._nmanifolds)])

    def exp(self, X, U):
        return [self._manifolds[k].exp(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    def retr(self, X, U):
        return [self._manifolds[k].retr(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    def log(self, X, U):
        return _TangentVector([self._manifolds[k].log(X[k], U[k])
                              for k in range(0, self._nmanifolds)])

    def rand(self):
        return [self._manifolds[k].rand()
                for k in range(0, self._nmanifolds)]

    def randvec(self, X):
        return _TangentVector([1/np.sqrt(self._nmanifolds) *
                              self._manifolds[k].randvec(X[k])
                              for k in range(0, self._nmanifolds)])

    def transp(self, X1, X2, G):
        return _TangentVector([self._manifolds[k].transp(X1[k], X2[k], G[k])
                               for k in range(0, self._nmanifolds)])

    def pairmean(self, X, Y):
        return [self._manifolds[k].pairmean(X[k], Y[k])
                for k in range(0, self._nmanifolds)]

    def zerovec(self, X):
        return _TangentVector([self._manifolds[k].zerovec(X[k])
                              for k in range(0, self._nmanifolds)])


class _TangentVector(list):
    def __add__(self, other):
        assert len(self) == len(other)
        return _TangentVector([self[k] + other[k] for k in range(len(self))])

    def __sub__(self, other):
        assert len(self) == len(other)
        return _TangentVector([self[k] - other[k] for k in range(len(self))])

    def __mul__(self, other):
        return _TangentVector([other * val for val in self])

    __rmul__ = __mul__

    def __div__(self, other):
        return _TangentVector([val / other for val in self])

    def __neg__(self):
        return _TangentVector([-val for val in self])
