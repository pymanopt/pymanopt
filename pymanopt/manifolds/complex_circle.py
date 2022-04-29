import numpy as np

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class ComplexCircle(EuclideanEmbeddedSubmanifold):
    """Manifold of unit-modulus complex numbers.

    Description of vectors z in C^n (complex) such that each component z(i)
    has unit modulus. The manifold structure is the Riemannian submanifold
    structure from the embedding space R^2 x ... x R^2, i.e., the complex
    circle is identified with the unit circle in the real plane.
    """

    def __init__(self, dimension=1):
        self._dimension = dimension
        if dimension == 1:
            name = "Complex circle S^1"
        else:
            name = f"Product manifold of complex circles (S^1)^{dimension}"
        super().__init__(name, dimension)

    def inner(self, point, tangent_vector_a, tangent_vector_b):
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

    tangent = projection

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        return self.projection(
            point,
            euclidean_hvp
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

    def log(self, x1, x2):
        v = self.projection(x1, x2 - x1)
        abs_v = np.abs(v)
        di = np.arccos((x1.conj() * x2).real)
        factors = di / abs_v
        factors[di <= 1e-6] = 1
        return v * factors

    def random_point(self):
        dimension = self._dimension
        return self._normalize(
            np.random.randn(dimension) + 1j * np.random.randn(dimension)
        )

    def random_tangent_vector(self, point):
        tangent_vector = np.random.randn(self._dimension) * 1j * point
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def pair_mean(self, point_a, point_b):
        return self._normalize(point_a + point_b)

    def zero_vector(self, point):
        return np.zeros(self._dimension)

    @staticmethod
    def _normalize(point):
        """Normalize entries of array by their absolute values."""
        return point / np.abs(point)
