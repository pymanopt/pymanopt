import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools import ndarraySequenceMixin


class Product(Manifold):
    """Product manifold, i.e., the cartesian product of multiple manifolds."""

    class _TangentVector(list, ndarraySequenceMixin):
        def __repr__(self):
            return "{:s}: {}".format(
                self.__class__.__name__, super().__repr__())

        def __add__(self, other):
            assert len(self) == len(other)
            return self.__class__(
                [v + other[k] for k, v in enumerate(self)])

        def __sub__(self, other):
            assert len(self) == len(other)
            return self.__class__(
                [v - other[k] for k, v in enumerate(self)])

        def __mul__(self, other):
            return self.__class__([s / other for s in self])

        __rmul__ = __mul__

        def __truediv__(self, other):
            return self.__class__([s / other for s in self])

        def __neg__(self):
            return self.__class__([-s for s in self])

    def __init__(self, *manifolds):
        if len(manifolds) == 0:
            raise ValueError("At least one manifold required")
        for manifold in manifolds:
            if not isinstance(manifold, Manifold):
                raise ValueError(
                    "Unsupport manifold of type '{}'".format(type(manifold)))
            if isinstance(manifold, Product):
                raise ValueError("Nested product manifolds are not supported")
        self._manifolds = tuple(manifolds)
        name = ("Product manifold: {:s}".format(
                " x ".join([str(man) for man in manifolds])))
        dimension = np.sum([manifold.dim for manifold in manifolds])
        point_layout = tuple(manifold.point_layout for manifold in manifolds)
        super().__init__(name, dimension, point_layout=point_layout)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            if key == "manifolds":
                raise AttributeError("Cannot override 'manifolds' attribute")
        super().__setattr__(key, value)

    @property
    def typicaldist(self):
        return np.sqrt(np.sum([manifold.typicaldist ** 2
                               for manifold in self._manifolds]))

    def inner(self, X, G, H):
        return np.sum([manifold.inner(X[k], G[k], H[k])
                       for k, manifold in enumerate(self._manifolds)])

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        return np.sqrt(np.sum([manifold.dist(X[k], Y[k]) ** 2
                               for k, manifold in enumerate(self._manifolds)]))

    def proj(self, X, U):
        return self._TangentVector(
            [manifold.proj(X[k], U[k])
             for k, manifold in enumerate(self._manifolds)])

    def tangent(self, X, U):
        return self._TangentVector(
            [manifold.tangent(X[k], U[k])
             for k, manifold in enumerate(self._manifolds)])

    def egrad2rgrad(self, X, U):
        return self._TangentVector(
            [manifold.egrad2rgrad(X[k], U[k])
             for k, manifold in enumerate(self._manifolds)])

    def ehess2rhess(self, X, egrad, ehess, H):
        return self._TangentVector(
            [manifold.ehess2rhess(X[k], egrad[k], ehess[k], H[k])
             for k, manifold in enumerate(self._manifolds)])

    def exp(self, X, U):
        return [manifold.exp(X[k], U[k])
                for k, manifold in enumerate(self._manifolds)]

    def retr(self, X, U):
        return [manifold.retr(X[k], U[k])
                for k, manifold in enumerate(self._manifolds)]

    def log(self, X, U):
        return self._TangentVector(
            [manifold.log(X[k], U[k])
             for k, manifold in enumerate(self._manifolds)])

    def rand(self):
        return [manifold.rand() for manifold in self._manifolds]

    def randvec(self, X):
        scale = len(self._manifolds) ** (-1/2)
        return self._TangentVector(
            [scale * manifold.randvec(X[k])
             for k, manifold in enumerate(self._manifolds)])

    def transp(self, X1, X2, G):
        return self._TangentVector(
            [manifold.transp(X1[k], X2[k], G[k])
             for k, manifold in enumerate(self._manifolds)])

    def pairmean(self, X, Y):
        return [manifold.pairmean(X[k], Y[k])
                for k, manifold in enumerate(self._manifolds)]

    def zerovec(self, X):
        return self._TangentVector(
            [manifold.zerovec(X[k])
             for k, manifold in enumerate(self._manifolds)])
