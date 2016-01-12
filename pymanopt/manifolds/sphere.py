import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from manifold import Manifold


class Sphere(Manifold):
    """
    Manifold of n-by-m real matrices of unit Frobenius norm. By default, m =
    1, which corresponds to the unit sphere in R^n. The metric is such that the
    sphere is a Riemannian submanifold of the space of n-by-m matrices with the
    usual trace inner product, i.e., the usual metric.
    """
    def __init__(self, n, m=1):
        self._n = n
        self._m = m

        if m == 1:
            self._name = "Sphere S^{:d}".format(n - 1)
        else:
            self._name = "Unit F-norm {:d}x{:d} matrices".format(n, m)

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._n * self._m - 1

    @property
    def typicaldist(self):
        return np.pi

    def inner(self, X, U, V):
        return float(np.tensordot(np.asmatrix(U), np.asmatrix(V)))

    def norm(self, X, U):
        return la.norm(U, "fro")

    def dist(self, U, V):
        return np.arccos(self.inner(None, U, V)).real

    def proj(self, X, H):
        return H - self.inner(None, X, H) * X

    tangent = proj

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, U):
        return self.proj(X, ehess) - self.inner(None, X, ehess) * U

    def exp(self, X, U, t=1):
        tU = t * U
        norm_tU = self.norm(None, tU)
        return X * np.cos(norm_tU) + tU * np.sin(norm_tU) / norm_tU

    def retr(self, X, U, t=1):
        Y = X + t * U
        return self._normalize(Y)

    def log(self, X, Y):
        proj = self.proj(X, Y - X)
        dist = self.dist(X, Y)
        # If the two points are "far apart", correct the norm.
        if dist > 1e-6:
            proj *= dist / self.norm(None, proj)
        return proj

    def rand(self):
        Y = rnd.randn(self._n, self._m)
        return self._normalize(Y)

    def randvec(self, X):
        H = rnd.randn(self._n, self._m)
        proj = self.proj(X, H)
        return self._normalize(proj)

    def transp(self, X, Y, U):
        return self.proj(Y, U)

    def pairmean(self, X, Y):
        return self._normalize(X + y)

    def _normalize(self, X):
        """
        Return a Frobenius-normalized version of the point X in the ambient
        space.
        """
        return X / self.norm(None, X)

