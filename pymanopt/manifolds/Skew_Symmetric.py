from __future__ import division

import numpy as np
import numpy.random as rnd

from pymanopt.manifolds import Euclidean
from pymanopt.tools.multi import multiskew


class Skew_Symmetric(Euclidean):
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

        self._dim = .5 * k * n * (n - 1)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    def proj(self, X, U):
        return multiskew(U)

    def egrad2rgrad(self, X, U):
        return multiskew(U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return multiskew(ehess)

    def rand(self):
        return multiskew(rnd.randn(*self._shape))

    def randvec(self, X):
        G = self.rand()
        return multiskew(G / self.norm(X, G))
