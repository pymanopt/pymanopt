import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class _Sphere(EuclideanEmbeddedSubmanifold):
    """Base class for tensors with unit Frobenius norm."""

    def __init__(self, *shape, name, dimension):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        self._shape = shape
        super().__init__(name, dimension)

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

    def weingarten(self, X, U, V):
        return -self.inner(X, X, V) * U

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

    def zerovec(self, X):
        return np.zeros(self._shape)

    def _normalize(self, X):
        """
        Return a Frobenius-normalized version of the point X in the ambient
        space.
        """
        return X / self.norm(None, X)


class Sphere(_Sphere):
    """Manifold of shape n1 x n2 x ... x nk tensors with unit 2-norm. The
    metric is such that the sphere is a Riemannian submanifold of Euclidean
    space.

    Notes
    -----
    The implementation of the Weingarten map is taken from [1]_.

    References
    ----------
    .. [1] Absil, P-A., Robert Mahony, and Jochen Trumpf. "An extrinsic look at
       the Riemannian Hessian." International Conference on Geometric Science
       of Information. Springer, Berlin, Heidelberg, 2013.
    """

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            name = "Sphere manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            name = "Sphere manifold of {}x{} matrices".format(*shape)
        else:
            name = "Sphere manifold of shape " + str(shape) + " tensors"
        dimension = np.prod(shape) - 1
        super().__init__(*shape, name=name, dimension=dimension)


class _SphereSubspaceIntersectionManifold(_Sphere):
    def __init__(self, projector, name, dimension):
        m, n = projector.shape
        assert m == n, "projection matrix is not square"
        if dimension == 0:
            warnings.warn(
                "Intersected subspace is 1-dimensional! The manifold '{:s}' "
                "therefore has dimension 0 as it only consists of isolated "
                "points".format(self._get_class_name()))
        self._subspace_projector = projector
        super().__init__(n, name=name, dimension=dimension)

    def _validate_span_matrix(self, U):
        if len(U.shape) != 2:
            raise ValueError("Input array must be 2-dimensional")
        num_rows, num_columns = U.shape
        if num_rows < num_columns:
            raise ValueError(
                "The span matrix cannot have fewer rows than columns")

    def proj(self, X, H):
        Y = super().proj(X, H)
        return self._subspace_projector.dot(Y)

    def rand(self):
        X = super().rand()
        return self._normalize(self._subspace_projector.dot(X))

    def randvec(self, X):
        Y = super().randvec(X)
        return self._normalize(self._subspace_projector.dot(Y))


class SphereSubspaceIntersection(_SphereSubspaceIntersectionManifold):
    """Manifold of n-dimensional unit 2-norm vectors intersecting the
    r-dimensional subspace of R^n spanned by the columns of the matrix U. This
    implementation is based on spheresubspacefactory.m from the Manopt MATLAB
    package.
    """

    def __init__(self, U):
        self._validate_span_matrix(U)
        m = U.shape[0]
        Q, _ = la.qr(U)
        projector = Q.dot(Q.T)
        subspace_dimension = la.matrix_rank(projector)
        name = ("Sphere manifold of {}-dimensional vectors intersecting a "
                "{}-dimensional subspace".format(m, subspace_dimension))
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)


class SphereSubspaceComplementIntersection(
        _SphereSubspaceIntersectionManifold):
    """Manifold of n-dimensional unit 2-norm vectors which are orthogonal to
    the r-dimensional subspace of R^n spanned by columns of the matrix U. This
    implementation is based on spheresubspacefactory.m from the Manopt MATLAB
    package.
    """

    def __init__(self, U):
        self._validate_span_matrix(U)
        m = U.shape[0]
        Q, _ = la.qr(U)
        projector = np.eye(m) - Q.dot(Q.T)
        subspace_dimension = la.matrix_rank(projector)
        name = ("Sphere manifold of {}-dimensional vectors orthogonal "
                "to a {}-dimensional subspace".format(m, subspace_dimension))
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)
