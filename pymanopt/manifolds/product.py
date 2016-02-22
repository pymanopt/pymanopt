import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


def listpoint(func):
    def decorated(*args, **kwargs):
        return ListPoint(func(*args, **kwargs))
    return decorated


class ListPoint(list):
    def __rmul__(self, other):
        return ListPoint([other * val for val in self])

    def __neg__(self):
        return ListPoint([-val for val in self])


class Product(Manifold):
    """
    Product manifold, i.e. the cartesian product of multiple manifolds.
    """
    def __init__(self, manifolds):
        self._manifolds = manifolds
        self._nmanifolds = len(manifolds)
        self._name = ("Product manifold: {:s}".format(
                          " X ".join([man.name for man in manifolds])))

    @property
    def name(self):
        return self._name

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
        return [self._manifolds[k].proj(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    @listpoint
    def egrad2rgrad(self, X, U):
        return [self._manifolds[k].egrad2rgrad(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    @listpoint
    def ehess2rhess(self, X, egrad, ehess, H):
        return [self._manifolds[k].ehess2rhess(X[k], egrad[k], ehess[k], H[k])
                for k in range(0, self._nmanifolds)]

    @listpoint
    def exp(self, X, U):
        return [self._manifolds[k].exp(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    def retr(self, X, U):
        return [self._manifolds[k].retr(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    @listpoint
    def log(self, X, U):
        return [self._manifolds[k].log(X[k], U[k])
                for k in range(0, self._nmanifolds)]

    def rand(self):
        return [self._manifolds[k].rand()
                for k in range(0, self._nmanifolds)]

    @listpoint
    def randvec(self, X):
        return [1/np.sqrt(self._nmanifolds) *
                self._manifolds[k].randvec(X[k])
                for k in range(0, self._nmanifolds)]

    @listpoint
    def transp(self, X1, X2, G):
        return [self._manifolds[k].transp(X1[k], X2[k], G[k])
                for k in range(0, self._nmanifolds)]

    def pairmean(self, X, Y):
        return [self._manifolds[k].pairmean(X[k], Y[k])
                for k in range(0, self._nmanifolds)]
