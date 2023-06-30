import numpy as np
from numpy import linalg as la
from numpy import random as rnd

from pymanopt.manifolds.manifold import RiemannianSubmanifold


class _ComplexEuclidean(RiemannianSubmanifold):
    """Shared base class for subspace manifolds of Euclidean space."""

    def __init__(self, name, dimension, *shape):
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim / 2)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.real(
            np.tensordot(
                tangent_vector_a.conj(),
                tangent_vector_b,
                axes=tangent_vector_a.ndim,
            )
        )

    def norm(self, point, tangent_vector):
        return la.norm(tangent_vector)

    def dist(self, point_a, point_b):
        return la.norm(point_a - point_b)

    def projection(self, point, vector):
        return vector

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return euclidean_hessian

    def exp(self, point, tangent_vector):
        return point + tangent_vector

    retr = exp

    def log(self, point_a, point_b):
        return point_b - point_a

    def random_point(self):
        return rnd.randn(*self._shape) + 1j * rnd.randn(*self._shape)

    def random_tangent_vector(self, point):
        Y = self.random_point()
        return Y / self.norm(point, Y)

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pair_mean(self, point_a, point_b):
        return (point_a + point_b) / 2

    def zero_vector(self, point):
        return np.zeros(self._shape, dtype=np.complex)


class ComplexEuclidean(_ComplexEuclidean):
    """Complex Euclidean manifold of shape n1 x n2 x ... x nk tensors.

    Useful for unconstrained optimization problems or
    for unconstrained hyperparameters, as part of
    a product manifold.

    Examples:
    Create a manifold of vectors of length n:
    manifold = ComplexEuclidean(n)

    Create a manifold of m x n matrices:
    manifold = ComplexEuclidean(m, n)
    """

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters")
        if len(shape) == 1:
            name = f"Euclidean manifold of {shape[0]}-vectors"
        elif len(shape) == 2:
            name = f"Euclidean manifold of {shape[0]}x{shape[1]} matrices"
        else:
            name = f"Euclidean manifold of shape {shape} tensors"
        dimension = 2 * np.prod(shape)
        super().__init__(name, dimension, *shape)
