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
        """Return Frobenius-normalized version of X in ambient space."""
        return X / self.norm(None, X)


class Sphere(_Sphere):
    r"""The sphere manifold.

    Manifold of shape :math:`n_1 \times n_2 \times \ldots \times n_k` tensors
    with unit 2-norm.
    The metric is such that the sphere is a Riemannian submanifold of Euclidean
    space.

    Notes:
        The implementation of the Weingarten map is taken from [AMT2013]_.
    """

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            (n1,) = shape
            name = f"Sphere manifold of {n1}-vectors"
        elif len(shape) == 2:
            n1, n2 = shape
            name = f"Sphere manifold of {n1}x{n2} matrices"
        else:
            name = f"Sphere manifold of shape {shape} tensors"
        dimension = np.prod(shape) - 1
        super().__init__(*shape, name=name, dimension=dimension)


class _SphereSubspaceIntersectionManifold(_Sphere):
    def __init__(self, projector, name, dimension):
        m, n = projector.shape
        assert m == n, "projection matrix is not square"
        if dimension == 0:
            warnings.warn(
                "Intersected subspace is 1-dimensional. The manifold "
                "therefore has dimension 0 as it only consists of isolated "
                "points"
            )
        self._subspace_projector = projector
        super().__init__(n, name=name, dimension=dimension)

    def _validate_span_matrix(self, U):
        if len(U.shape) != 2:
            raise ValueError("Input array must be 2-dimensional")
        num_rows, num_columns = U.shape
        if num_rows < num_columns:
            raise ValueError(
                "The span matrix cannot have fewer rows than columns"
            )

    def proj(self, X, H):
        Y = super().proj(X, H)
        return self._subspace_projector @ Y

    def rand(self):
        X = super().rand()
        return self._normalize(self._subspace_projector @ X)

    def randvec(self, X):
        Y = super().randvec(X)
        return self._normalize(self._subspace_projector @ Y)


class SphereSubspaceIntersection(_SphereSubspaceIntersectionManifold):
    r"""Sphere-subspace intersection manifold.

    Manifold of n-dimensional unit 2-norm vectors intersecting the
    :math:`r`-dimensional subspace of :math:`\R^n` spanned by the columns of
    the matrix ``U`` of size :math:`n \times r`.
    """

    def __init__(self, U):
        self._validate_span_matrix(U)
        m = U.shape[0]
        Q, _ = la.qr(U)
        projector = Q @ Q.T
        subspace_dimension = la.matrix_rank(projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors intersecting a "
            f"{subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)


class SphereSubspaceComplementIntersection(
    _SphereSubspaceIntersectionManifold
):
    r"""Sphere-subspace compliment intersection manifold.

    Manifold of n-dimensional unit 2-norm vectors which are orthogonal to
    the :math:`r`-dimensional subspace of :math:`\R^n` spanned by columns of
    the matrix ``U``.
    """

    def __init__(self, U):
        self._validate_span_matrix(U)
        m = U.shape[0]
        Q, _ = la.qr(U)
        projector = np.eye(m) - Q @ Q.T
        subspace_dimension = la.matrix_rank(projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors orthogonal "
            f"to a {subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)
