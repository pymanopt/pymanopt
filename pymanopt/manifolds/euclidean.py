import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class Euclidean(Manifold):
    """
    Euclidean manifold, i.e. the euclidean space of m-by-n matrices
    equipped with the Frobenius distance and trace inner product.
    Use for solving unconstrained problems with pymanopt.
    """

    def __init__(self, *shape):
        self._shape = shape
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        elif len(shape) == 1:
            self._name = "Euclidean manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            self._name = ("Euclidean manifold of {}x{} "
                           "matricies").format(*shape)
        else:
            self._name = ("Euclidean manifold of shape " + str(shape) +
                           " tensors")

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return np.prod(self._shape)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, X, G, H):
        return float(np.tensordot(G, H, axes=G.ndim))

    def norm(self, X, G):
        return la.norm(G)

    def dist(self, X, Y):
        return la.norm(X-Y)

    def proj(self, X, U):
        return U

    def egrad2rgrad(self, X, U):
        return U

    def ehess2rhess(self, X, egrad, ehess, H):
        return ehess

    def exp(self, X, U):
        return X+U

    retr = exp

    def log(self, X, Y):
        return Y-X

    def rand(self):
        return rnd.randn(*self._shape)

    def randvec(self, X):
        Y = self.rand()
        return Y / self.norm(X, Y)

    def transp(self, X1, X2, G):
        return G

    def pairmean(self, X, Y):
        return .5*(X+Y)
