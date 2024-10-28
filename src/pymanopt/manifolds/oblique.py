from pymanopt.backends import Backend, DummyBackendSingleton
from pymanopt.manifolds.manifold import RiemannianSubmanifold


class Oblique(RiemannianSubmanifold):
    r"""Manifold of matrices with unit-norm columns.

    The oblique manifold deals with matrices of size ``m x n`` such that each
    column has unit Euclidean norm, i.e., is a point on the unit sphere in
    :math:`\R^m`.
    The metric is such that the oblique manifold is a Riemannian submanifold of
    the space of ``m x n`` matrices with the usual trace inner product.

    Args:
        m: The number of rows of each matrix.
        n: The number of columns of each matrix.
    """

    def __init__(
        self,
        m: int,
        n: int,
        backend: Backend = DummyBackendSingleton,
    ):
        self._m = m
        self._n = n
        name = f"Oblique manifold OB({m}, {n})"
        dimension = (m - 1) * n
        super().__init__(name, dimension, backend=backend)

    @property
    def typical_dist(self):
        return self.backend.pi * self.backend.sqrt(self._n)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self.backend.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return self.backend.linalg_norm(tangent_vector)

    def dist(self, point_a, point_b):
        XY = self.backend.sum(point_a * point_b, 0)
        # XY[XY > 1] = 1
        XY = self.backend.where(XY > 1, 1, XY)
        return self.backend.linalg_norm(self.backend.arccos(XY))

    def projection(self, point, vector):
        return vector - point * (
            self.backend.sum(point * vector, 0)[self.backend.newaxis, :]
        )

    to_tangent_space = projection

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hessian)
        return PXehess - tangent_vector * (
            self.backend.sum(point * euclidean_gradient, 0)[
                self.backend.newaxis, :
            ]
        )

    def exp(self, point, tangent_vector):
        norm = self.backend.sqrt(self.backend.sum(tangent_vector**2, 0))[
            self.backend.newaxis, :
        ]
        target_point = point * self.backend.cos(
            norm
        ) + tangent_vector * self.backend.sinc(norm / self.backend.pi)
        return target_point

    def retraction(self, point, tangent_vector):
        return self._normalize_columns(point + tangent_vector)

    def log(self, point_a, point_b):
        vector = self.projection(point_a, point_b - point_a)
        distances = self.backend.arccos(self.backend.sum(point_a * point_b, 0))
        norms = self.backend.real(
            self.backend.sqrt(self.backend.sum(vector**2, 0))
        )
        # Try to avoid zero-division when both distances and norms are almost
        # zero.
        epsilon = self.backend.eps()
        factors = (distances + epsilon) / (norms + epsilon)
        return vector * factors

    def random_point(self):
        return self._normalize_columns(
            self.backend.random_normal(size=(self._m, self._n))
        )

    def random_tangent_vector(self, point):
        vector = self.backend.random_normal(size=point.shape)
        tangent_vector = self.projection(point, vector)
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def pair_mean(self, point_a, point_b):
        return self._normalize_columns(point_a + point_b)

    def zero_vector(self, point):
        return self.backend.zeros((self._m, self._n))

    def _normalize_columns(self, array):
        return (
            array
            / self.backend.linalg_norm(array, axis=0)[self.backend.newaxis, :]
        )
