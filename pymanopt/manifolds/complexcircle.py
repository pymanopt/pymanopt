from __future__ import division

import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class ComplexCircle(Manifold):
    """
    The manifold of complex numbers with unit-modulus.

    Description of vectors z in C^n (complex) such that each component z(i)
    has unit modulus. The manifold structure is the Riemannian submanifold
    structure from the embedding space R^2 x ... x R^2, i.e., the complex
    circle is identified with the unit circle in the real plane.
    """

    def __init__(self, n=1):
        if n == 1:
            self._name = "Complex circle S^1"
        else:
            self._name = "Complex circle (S^1)^{:d}".format(n)
        self._n = n

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._n

    def inner(self, z, v, w):
        return v.conj().dot(w).real

    def norm(self, x, v):
        return la.norm(v)

    def dist(self, x, y):
        return la.norm(np.arccos((x.conj() * y).real))

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._n)

    def proj(self, z, u):
        return u - (u.conj() * z).real * z

    tangent = proj

    def ehess2rhess(self, z, egrad, ehess, zdot):
        return self.proj(z, (z * egrad.conj()).real * zdot)

    def exp(self, z, v):
        y = np.zeros(self._n)
        abs_v = np.abs(v)
        mask = abs_v > 0
        not_mask = np.logical_not(mask)
        y[mask] = (z[mask] * np.cos(abs_v[mask]) +
                   v[mask] * (np.sin(abs_v[mask]) / abs_v[mask]))
        y[not_mask] = z[not_mask]
        return y

    def retr(self, z, v):
        return np.sign(z + v)

    def log(self, x1, x2):
        v = self.proj(x1, x2 - x1)
        abs_v = np.abs(v)
        di = np.arccos((x1.conj() * x2).real)
        factors = di / abs_v
        factors[di <= 1e-6] = 1
        return v * factors

    def rand(self):
        n = self._n
        return np.sign(rnd.randn(n) + 1j * rnd.randn(n))

    def randvec(self, z):
        v = rnd.randn(self._n) * (1j * z)
        return v / self.norm(z, v)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def pairmean(self, z1, z2):
        return np.sign(z1 + z2)

    # XXX: Do we need to supply vec(x, u_mat) and mat(x, u_vec)?
