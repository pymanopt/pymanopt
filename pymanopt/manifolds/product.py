from __future__ import division

import numpy as np

from pymanopt.manifolds.manifold import Manifold


class Product(Manifold):
    """
    Product manifold, i.e. the cartesian product of multiple manifolds.
    """

    def __init__(self, manifolds):
        self._manifolds = manifolds

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
        return np.sum([man.inner(X[k], G[k], H[k])
                       for k, man in enumerate(self._manifolds)])

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        return np.sqrt(np.sum([man.dist(X[k], Y[k])**2
                               for k, man in enumerate(self._manifolds)]))

    def proj(self, X, U):
        return _TangentVector([man.proj(X[k], U[k])
                               for k, man in enumerate(self._manifolds)])

    def egrad2rgrad(self, X, U):
        return _TangentVector([man.egrad2rgrad(X[k], U[k])
                               for k, man in enumerate(self._manifolds)])

    def ehess2rhess(self, X, egrad, ehess, H):
        return _TangentVector([man.ehess2rhess(X[k], egrad[k], ehess[k], H[k])
                               for k, man in enumerate(self._manifolds)])

    def exp(self, X, U):
        return [man.exp(X[k], U[k]) for k, man in enumerate(self._manifolds)]

    def retr(self, X, U):
        return [man.retr(X[k], U[k]) for k, man in enumerate(self._manifolds)]

    def log(self, X, U):
        return _TangentVector([man.log(X[k], U[k])
                               for k, man in enumerate(self._manifolds)])

    def rand(self):
        return [man.rand() for man in self._manifolds]

    def randvec(self, X):
        scale = len(self._manifolds) ** (-1/2)
        return _TangentVector([scale * man.randvec(X[k])
                               for k, man in enumerate(self._manifolds)])

    def transp(self, X1, X2, G):
        return _TangentVector([man.transp(X1[k], X2[k], G[k])
                               for k, man in enumerate(self._manifolds)])

    def pairmean(self, X, Y):
        return [man.pairmean(X[k], Y[k])
                for k, man in enumerate(self._manifolds)]

    def zerovec(self, X):
        return _TangentVector([man.zerovec(X[k])
                               for k, man in enumerate(self._manifolds)])


class _TangentVector(list):
    # These attributes ensure that multiplication of _TangentVector instances
    # with scalar np.float64 variables and the like is commutative. By default,
    # multiplication of an unknown object from the right with a numpy data type
    # causes numpy to attempt vectorization for subsequent array broadcasting.
    # These two attributes essentially cause the output type of binary
    # operations involving _TangentVector and ndarray to remain _TangentVector.
    # See
    #     https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    #     https://github.com/pymanopt/pymanopt/issues/49
    # for details.
    __array_priority__ = 1000
    __array_ufunc__ = None  # Available since numpy 1.13

    def __repr__(self):
        repr_ = super(_TangentVector, self).__repr__()
        return "TangentVector: " + repr_

    def __add__(self, other):
        assert len(self) == len(other)
        return _TangentVector([v + other[k] for k, v in enumerate(self)])

    def __sub__(self, other):
        assert len(self) == len(other)
        return _TangentVector([v - other[k] for k, v in enumerate(self)])

    def __mul__(self, other):
        return _TangentVector([other * val for val in self])

    __rmul__ = __mul__

    def __div__(self, other):
        return _TangentVector([val / other for val in self])

    def __neg__(self):
        return _TangentVector([-val for val in self])
