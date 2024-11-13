import math
import warnings

from pymanopt.backends import Backend, DummyBackendSingleton
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools import extend_docstring


class _SphereBase(RiemannianSubmanifold):
    def __init__(
        self,
        *shape,
        name,
        dimension,
        backend: Backend = DummyBackendSingleton,
    ):
        if len(shape) == 0:
            raise TypeError("Need at least one dimension.")
        self._shape = shape
        super().__init__(name, dimension, backend=backend)

    @property
    def typical_dist(self):
        return self.backend.pi

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self.backend.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return self.backend.linalg_norm(tangent_vector)

    def dist(self, point_a, point_b):
        inner = max(min(self.inner_product(point_a, point_a, point_b), 1), -1)
        return self.backend.arccos(inner)

    def projection(self, point, vector):
        return vector - self.inner_product(point, point, vector) * point

    to_tangent_space = projection

    def weingarten(self, point, tangent_vector, normal_vector):
        return (
            -self.inner_product(point, point, normal_vector) * tangent_vector
        )

    def exp(self, point, tangent_vector):
        norm = self.norm(point, tangent_vector)
        return point * self.backend.cos(
            norm
        ) + tangent_vector * self.backend.sinc(norm / self.backend.pi)

    def retraction(self, point, tangent_vector):
        return self._normalize(point + tangent_vector)

    def log(self, point_a, point_b):
        vector = self.projection(point_a, point_b - point_a)
        distance = self.dist(point_a, point_b)
        epsilon = self.backend.eps()
        factor = (distance + epsilon) / (self.norm(point_a, vector) + epsilon)
        return factor * vector

    def random_point(self):
        point = self.backend.random_normal(size=self._shape)
        return self._normalize(point)

    def random_tangent_vector(self, point):
        vector = self.backend.random_normal(size=self._shape)
        return self._normalize(self.projection(point, vector))

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def pair_mean(self, point_a, point_b):
        return self._normalize(point_a + point_b)

    def zero_vector(self, point):
        return self.backend.zeros(self._shape)

    def _normalize(self, array):
        return array / self.backend.linalg_norm(array)


DOCSTRING_NOTE = """
    Note:
        The Weingarten map is taken from [AMT2013]_.
"""


@extend_docstring(DOCSTRING_NOTE)
class Sphere(_SphereBase):
    r"""The sphere manifold.

    Manifold of shape :math:`n_1 \times \ldots \times n_k` tensors with unit
    Euclidean norm.
    The norm is understood as the :math:`\ell_2`-norm of :math:`\E =
    \R^{\sum_{i=1}^k n_i}` after identifying :math:`\R^{n_1 \times \ldots
    \times n_k}` with :math:`\E`.
    The metric is the one inherited from the usual Euclidean inner product that
    induces :math:`\norm{\cdot}_2` on :math:`\E` such that the manifold forms a
    Riemannian submanifold of Euclidean space.

    Args:
        shape: The shape of tensors.
    """

    def __init__(
        self,
        *shape: int,
        backend: Backend = DummyBackendSingleton,
    ):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            (n,) = shape
            name = f"Sphere manifold of {n}-vectors"
        elif len(shape) == 2:
            m, n = shape
            name = f"Sphere manifold of {m}x{n} matrices"
        else:
            name = f"Sphere manifold of shape {shape} tensors"
        dimension = math.prod(shape) - 1
        super().__init__(
            *shape, name=name, dimension=dimension, backend=backend
        )


class _SphereSubspaceIntersectionManifold(_SphereBase):
    def __init__(self, name, dimension, matrix, subspace_projector, backend):
        m, n = subspace_projector.shape
        assert m == n, "projection matrix is not square"
        if dimension == 0:
            warnings.warn(
                "Intersected subspace is 1-dimensional. The manifold "
                "therefore has dimension 0 as it only consists of isolated "
                "points"
            )
        self._matrix = matrix
        self._validate_span_matrix(matrix, backend)
        self._subspace_projector = subspace_projector
        super().__init__(n, name=name, dimension=dimension, backend=backend)

    def _validate_span_matrix(self, matrix, backend: Backend):
        if not isinstance(matrix, backend.array_t):
            raise ValueError(
                f"The span matrix must be of type {backend.array_t}"
            )
        if len(matrix.shape) != 2:
            raise ValueError("Input array must be 2-dimensional")
        num_rows, num_columns = matrix.shape
        if num_rows < num_columns:
            raise ValueError(
                "The span matrix cannot have fewer rows than columns"
            )

    def projection(self, point, vector):
        return self._subspace_projector @ super().projection(point, vector)

    def random_point(self):
        return self._normalize(
            self._subspace_projector @ super().random_point()
        )

    def random_tangent_vector(self, point):
        return self._normalize(
            self._subspace_projector @ super().random_tangent_vector(point)
        )


@extend_docstring(DOCSTRING_NOTE)
class SphereSubspaceIntersection(_SphereSubspaceIntersectionManifold):
    r"""Sphere-subspace intersection manifold.

    Manifold of :math:`n`-dimensional vectors with unit :math:`\ell_2`-norm
    intersecting an :math:`r`-dimensional subspace of :math:`\R^n`.
    The subspace is represented by a matrix of size ``n x r`` whose columns
    span the subspace.

    Args:
        matrix: Matrix whose columns span the intersecting subspace.
    """

    def __init__(
        self,
        matrix,
        backend: Backend = NumpyBackend(),  # noqa: B008
    ):
        if backend is None:
            raise ValueError(
                f"A backend must always be specified for class {__class__.__name__}"
            )
        m = matrix.shape[0]
        q, _ = backend.linalg_qr(matrix)
        subspace_projector = q @ backend.transpose(q)
        subspace_dimension = backend.linalg_matrix_rank(subspace_projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors intersecting a "
            f"{subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(name, dimension, matrix, subspace_projector, backend)

    @RiemannianSubmanifold.backend.setter
    def _(self, backend: Backend):
        super().backend = backend
        self._matrix = backend.array(self._matrix)
        q, _ = backend.linalg_qr(self._matrix)
        self._subspace_projector = q @ self.backend.transpose(q)


@extend_docstring(DOCSTRING_NOTE)
class SphereSubspaceComplementIntersection(
    _SphereSubspaceIntersectionManifold
):
    r"""Sphere-subspace complement intersection manifold.

    Manifold of :math:`n`-dimensional vectors with unit :math:`\ell_2`-norm
    that are orthogonal to an :math:`r`-dimensional subspace of :math:`\R^n`.
    The subspace is represented by a matrix of size ``n x r`` whose columns
    span the subspace.

    Args:
        matrix: Matrix whose columns span the subspace.
    """

    def __init__(
        self,
        matrix,
        backend: Backend = NumpyBackend(),  # noqa: B008
    ):
        if backend is None:
            raise ValueError(
                f"A backend must always be specified for class {__class__.__name__}"
            )
        m = matrix.shape[0]
        q, _ = backend.linalg_qr(matrix)
        subspace_projector = backend.eye(m) - q @ backend.transpose(q)
        subspace_dimension = backend.linalg_matrix_rank(subspace_projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors orthogonal "
            f"to a {subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(name, dimension, matrix, subspace_projector, backend)

    @RiemannianSubmanifold.backend.setter
    def _(self, backend: Backend):
        super().backend = backend
        self._matrix = backend.array(self._matrix)
        q, _ = backend.linalg_qr(self._matrix)
        self._subspace_projector = backend.eye(
            self.dim
        ) - q @ self.backend.transpose(q)
