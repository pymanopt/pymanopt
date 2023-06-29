import numpy as np

from pymanopt.manifolds.manifold import RiemannianSubmanifold


class ComplexCircle(RiemannianSubmanifold):
    r"""Manifold of unit-modulus complex numbers.

    Manifold of complex vectors :math:`\vmz` in :math:`\C^n` such that each
    component :math:`z_i` has unit modulus :math:`\abs{z_i} = 1`.

    Args:
        n: The dimension of the underlying complex space.

    Note:
        The manifold structure is the Riemannian submanifold
        structure from the embedding space :math:`\R^2 \times \ldots \times
        \R^2`, i.e., the complex circle identified with the unit circle in the
        real plane.
    """

    def __init__(self, n=1):
        self._n = n
        if n == 1:
            name = "Complex circle S^1"
        else:
            name = f"Product manifold of complex circles (S^1)^{n}"
        super().__init__(name, n)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return (tangent_vector_a.conj() @ tangent_vector_b).real

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def dist(self, point_a, point_b):
        return np.linalg.norm(np.arccos((point_a.conj() * point_b).real))

    @property
    def typical_dist(self):
        return np.pi * np.sqrt(self._dimension)

    def projection(self, point, vector):
        return vector - (vector.conj() * point).real * point

    to_tangent_space = projection

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return self.projection(
            point,
            euclidean_hessian
            - (point * euclidean_gradient.conj()).real * tangent_vector,
        )

    def exp(self, point, tangent_vector):
        tangent_vector_abs = np.abs(tangent_vector)
        mask = tangent_vector_abs > 0
        not_mask = np.logical_not(mask)
        tangent_vector_new = np.zeros(self._dimension)
        tangent_vector_new[mask] = point[mask] * np.cos(
            tangent_vector_abs[mask]
        ) + tangent_vector[mask] * (
            np.sin(tangent_vector_abs[mask]) / tangent_vector_abs[mask]
        )
        tangent_vector_new[not_mask] = point[not_mask]
        return tangent_vector_new

    def retraction(self, point, tangent_vector):
        return self._normalize(point + tangent_vector)

    def log(self, point_a, point_b):
        v = self.projection(point_a, point_b - point_a)
        abs_v = np.abs(v)
        di = np.arccos((point_a.conj() * point_b).real)
        factors = di / abs_v
        factors[di <= 1e-6] = 1
        return v * factors

    def random_point(self):
        dimension = self._dimension
        return self._normalize(
            np.random.normal(size=dimension)
            + 1j * np.random.normal(size=dimension)
        )

    def random_tangent_vector(self, point):
        tangent_vector = np.random.normal(size=self._dimension) * 1j * point
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def pair_mean(self, point_a, point_b):
        return self._normalize(point_a + point_b)

    def zero_vector(self, point):
        return np.zeros(self._dimension)

    @staticmethod
    def _normalize(point):
        return point / np.abs(point)
