import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold


class Sphere(Manifold):
    """
    Manifold of shape n1 x n2 x ... x nk tensors with unit 2-norm. The
    metric is such that the sphere is a Riemannian submanifold of Euclidean
    space. This implementation is based on spherefactory.m from the Manopt
    MATLAB package.

    Examples:
    The 'usual' sphere S^2, the set of points lying on the surface of a ball in
    3D space:
    sphere = Sphere(3)
    """

    def __init__(self, *shape):
        self._shape = shape
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        elif len(shape) == 1:
            self._name = "Sphere manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            self._name = "Sphere manifold of {}x{} matrices".format(*shape)
        else:
            self._name = "Sphere manifold of shape " + str(shape) + " tensors"

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return np.prod(self._shape) - 1

    @property
    def typicaldist(self):
        return np.pi

    def inner(self, X, U, V):
        return float(np.tensordot(U, V, axes=U.ndim))

    def norm(self, X, U):
        return la.norm(U)

    def dist(self, U, V):
        # Make sure inner product is between -1 and 1
        inner = max(min(self.inner(None, U, V), 1), -1)
        return np.arccos(inner)

    def proj(self, X, H):
        return H - self.inner(None, X, H) * X

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, U):
        return self.proj(X, ehess) - self.inner(None, X, egrad) * U

    def exp(self, X, U):
        norm_U = self.norm(None, U)
        # Check that norm_U isn't too tiny. If very small then
        # sin(norm_U) / norm_U ~= 1 and retr is extremely close to exp.
        if norm_U > 1e-3:
            return X * np.cos(norm_U) + U * np.sin(norm_U) / norm_U
        else:
            return self.retr(X, U)

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
        Y = rnd.randn(*self._shape)
        return self._normalize(Y)

    def randvec(self, X):
        H = rnd.randn(*self._shape)
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
