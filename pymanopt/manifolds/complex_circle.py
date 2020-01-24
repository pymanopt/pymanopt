import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class ComplexCircle(EuclideanEmbeddedSubmanifold):
    """
    The manifold of complex numbers with unit-modulus.

    Description of vectors z in C^n (complex) such that each component z(i)
    has unit modulus. The manifold structure is the Riemannian submanifold
    structure from the embedding space R^2 x ... x R^2, i.e., the complex
    circle is identified with the unit circle in the real plane. This
    implementation is based on complexcirclefactory.m from the Manopt MATLAB
    package.
    """

    def __init__(self, dimension=1):
        self._dimension = dimension
        if dimension == 1:
            name = "Complex circle S^1"
        else:
            name = "Complex circle (S^1)^{:d}".format(dimension)
        super().__init__(name, dimension)

    def inner(self, z, v, w):
        return v.conj().dot(w).real

    def norm(self, x, v):
        return la.norm(v)

    def dist(self, x, y):
        return la.norm(np.arccos((x.conj() * y).real))

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._dimension)

    def proj(self, z, u):
        return u - (u.conj() * z).real * z

    tangent = proj

    def ehess2rhess(self, z, egrad, ehess, zdot):
        return self.proj(z, (z * egrad.conj()).real * zdot)

    def exp(self, z, v):
        y = np.zeros(self._dimension)
        abs_v = np.abs(v)
        mask = abs_v > 0
        not_mask = np.logical_not(mask)
        y[mask] = (z[mask] * np.cos(abs_v[mask]) +
                   v[mask] * (np.sin(abs_v[mask]) / abs_v[mask]))
        y[not_mask] = z[not_mask]
        return y

    def retr(self, z, v):
        return self._normalize(z + v)

    def log(self, x1, x2):
        v = self.proj(x1, x2 - x1)
        abs_v = np.abs(v)
        di = np.arccos((x1.conj() * x2).real)
        factors = di / abs_v
        factors[di <= 1e-6] = 1
        return v * factors

    def rand(self):
        dimension = self._dimension
        return self._normalize(
            rnd.randn(dimension) + 1j * rnd.randn(dimension))

    def randvec(self, z):
        v = rnd.randn(self._dimension) * (1j * z)
        return v / self.norm(z, v)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def pairmean(self, z1, z2):
        return self._normalize(z1 + z2)

    def zerovec(self, x):
        return np.zeros(self._dimension)

    @staticmethod
    def _normalize(x):
        """Normalize the entries of x element-wise by their absolute values."""
        return x / np.abs(x)
