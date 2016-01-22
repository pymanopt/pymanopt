import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class Sphere(Manifold):
    """
    Manifold of m-by-n real matrices of unit Frobenius norm. By default, n =
    1, which corresponds to the unit sphere in R^m. The metric is such that the
    sphere is a Riemannian submanifold of the space of m-by-n matrices with the
    usual trace inner product, i.e., the usual metric.
    """
    def __init__(self, m, n=1):
        self._m = m
        self._n = n

        if n == 1:
            self._name = "Sphere S^{:d}".format(m - 1)
        else:
            self._name = "Unit F-norm {:d}x{:d} matrices".format(m, n)

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
        return self.proj(X, ehess) - self.inner(None, X, egrad) * U

    def exp(self, X, U):
        norm_U = self.norm(None, U)
        return X * np.cos(norm_U) + U * np.sin(norm_U) / norm_U

    def retr(self, X, U):
        Y = X + U
        return self._normalize(Y)

    def log(self, X, Y):
        P = self.proj(X, Y - X)
        dist = self.dist(X, Y)
        # If the two points are "far apart", correct the norm.
        if dist > 1e-6:
            P *= dist / self.norm(None, P)
        return P

    def rand(self):
        Y = rnd.randn(self._m, self._n)
        return self._normalize(Y)

    def randvec(self, X):
        H = rnd.randn(*X.shape)
        P = self.proj(X, H)
        return self._normalize(P)

    def transp(self, X, Y, U):
        return self.proj(Y, U)

    def pairmean(self, X, Y):
        return self._normalize(X + Y)

    def _normalize(self, X):
        """
        Return a Frobenius-normalized version of the point X in the ambient
        space.
        """
        return X / self.norm(None, X)
