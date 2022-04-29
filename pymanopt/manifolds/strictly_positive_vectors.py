import numpy as np

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class StrictlyPositiveVectors(EuclideanEmbeddedSubmanifold):
    r"""Manifold of strictly positive vectors.

    Since :math:`((\R_{++})^n)^k` is isomorphic to the manifold of positive
    definite diagonal matrices the geometry is inherited from the geometry of
    positive definite matrices.
    """

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of strictly positive {n}-vectors"
        else:
            name = f"Product manifold of {k} strictly positive {n}-vectors"
        dimension = int(k * n)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        inv_x = 1.0 / point
        return np.sum(
            inv_x * tangent_vector_a * inv_x * tangent_vector_b,
            axis=0,
            keepdims=True,
        )

    def projection(self, point, tangent_vector):
        return tangent_vector

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner(point, tangent_vector, tangent_vector))

    def random_point(self):
        return np.random.uniform(low=1e-6, high=1, size=(self._n, self._k))

    def random_tangent_vector(self, point):
        vector = np.random.randn(self._n, self._k)
        return vector / self.norm(point, vector)

    def zero_vector(self, point):
        return np.zeros(self._n, self._k)

    def dist(self, point_a, point_b):
        return np.linalg.norm(
            np.log(point_a) - np.log(point_b), axis=0, keepdims=True
        )

    def egrad2rgrad(self, point, euclidean_gradient):
        return euclidean_gradient * point**2

    def exp(self, point, tangent_vector):
        return point * np.exp((1.0 / point) * tangent_vector)

    def retraction(self, point, tangent_vector):
        return point + tangent_vector

    def log(self, point_a, point_b):
        return point_a * np.log((1.0 / point_a) * point_b)
