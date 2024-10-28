import math

import numpy as np

from pymanopt.backends import Backend, DummyBackendSingleton
from pymanopt.manifolds.manifold import RiemannianSubmanifold


class _Euclidean(RiemannianSubmanifold):
    def __init__(
        self,
        name,
        dimension,
        *shape,
        backend: Backend = DummyBackendSingleton,
    ):
        self._shape = shape
        super().__init__(name, dimension, backend=backend)

    @property
    def typical_dist(self):
        return self.backend.sqrt(self.dim)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return float(
            self.backend.real(
                self.backend.tensordot(
                    self.backend.conjugate(tangent_vector_a),
                    tangent_vector_b,
                    axes=tangent_vector_a.ndim,
                )
            )
        )

    def norm(self, point, tangent_vector):
        return self.backend.linalg_norm(tangent_vector)

    def dist(self, point_a, point_b):
        return self.backend.linalg_norm(point_a - point_b)

    def projection(self, point, vector):
        return vector

    to_tangent_space = projection

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return euclidean_hessian

    def exp(self, point, tangent_vector):
        return point + tangent_vector

    retraction = exp

    def log(self, point_a, point_b):
        return point_b - point_a

    def random_point(self):
        return self.backend.random_normal(size=self._shape)

    def random_tangent_vector(self, point):
        tangent_vector = self.random_point()
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pair_mean(self, point_a, point_b):
        return (point_a + point_b) / 2

    def zero_vector(self, point):
        return self.backend.zeros(self._shape)


class Euclidean(_Euclidean):
    r"""Euclidean manifold.

    Args:
        shape: Shape of points on the manifold.

    Note:
        If ``shape == (n,)``, this is the manifold of vectors with the
        standard Euclidean inner product, i.e., :math:`\R^n`.
        For ``shape == (m, n)``, it corresponds to the manifold of ``m x n``
        matrices equipped with the standard trace inner product.
        For ``shape == (n1, n2, ..., nk)``, the class represents the manifold
        of tensors of shape ``n1 x n2 x ... x nk`` with the inner product
        corresponding to the usual tensor dot product.
    """

    def __init__(
        self,
        *shape: int,
        backend: Backend = DummyBackendSingleton,
    ):
        if len(shape) == 0:
            raise TypeError("Need shape parameters")
        if len(shape) == 1:
            (n1,) = shape
            name = f"Euclidean manifold of {n1}-vectors"
        elif len(shape) == 2:
            n1, n2 = shape
            name = f"Euclidean manifold of {n1}x{n2} matrices"
        else:
            name = f"Euclidean manifold of shape {shape} tensors"
        dimension = math.prod(shape)
        super().__init__(name, dimension, *shape, backend=backend)

        @RiemannianSubmanifold.backend.setter
        def _(self, backend: Backend):
            assert backend.is_dtype_real()
            super().backend = backend


class ComplexEuclidean(_Euclidean):
    r"""Complex Euclidean manifold.

    Args:
        shape: Shape of points on the manifold.

    Note:
        If ``shape == (n,)``, this is the manifold of vectors with the
        standard Euclidean inner product, i.e., :math:`\C^n`.
        For ``shape == (m, n)``, it corresponds to the manifold of ``m x n``
        matrices equipped with the standard trace inner product.
        For ``shape == (n1, n2, ..., nk)``, the class represents the manifold
        of tensors of shape ``n1 x n2 x ... x nk`` with the inner product
        corresponding to the usual tensor dot product.
    """

    IS_COMPLEX = True

    def __init__(self, *shape, backend: Backend = DummyBackendSingleton):
        if len(shape) == 0:
            raise TypeError("Need shape parameters")
        if len(shape) == 1:
            (n1,) = shape
            name = f"Complex Euclidean manifold of {n1}-vectors"
        elif len(shape) == 2:
            n1, n2 = shape
            name = f"Complex Euclidean manifold of {n1}x{n2} matrices"
        else:
            name = f"Complex Euclidean manifold of shape {shape} tensors"
        dimension = 2 * np.prod(shape)
        super().__init__(name, dimension, *shape, backend=backend)

    @RiemannianSubmanifold.backend.setter
    def _(self, backend: Backend):
        assert not backend.is_dtype_real()
        super().backend = backend

    def random_point(self):
        return self.backend.random_randn(
            *self._shape
        ) + 1j * self.backend.random_randn(*self._shape)

    def zero_vector(self, point):
        return np.zeros(self._shape, dtype=complex)


class Symmetric(_Euclidean):
    """(Product) manifold of symmetric matrices.

    Args:
        n: Number of rows and columns of matrices.
        k: Number of elements in the product manifold.

    Note:
        Manifold of ``n x n`` symmetric matrices as a Riemannian submanifold of
        Euclidean space.
        If ``k > 1`` then this is the product manifold of ``k`` symmetric ``n x
        n`` matrices represented as arrays of shape ``(k, n, n)``.
    """

    def __init__(
        self,
        n: int,
        k: int = 1,
        backend: Backend = DummyBackendSingleton,
    ):
        if k == 1:
            shape = (n, n)
            name = f"Manifold of {n}x{n} symmetric matrices"
        elif k > 1:
            shape = (k, n, n)
            name = f"Product manifold of {k} {n}x{n} symmetric matrices"
        else:
            raise ValueError(f"k must be an integer no less than 1, got {k}")
        dimension = int(k * n * (n + 1) / 2)
        super().__init__(name, dimension, *shape, backend=backend)

    def projection(self, point, vector):
        return self.backend.sym(vector)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return self.backend.sym(euclidean_hessian)

    def random_point(self):
        return self.backend.sym(self.backend.random_normal(size=self._shape))

    def random_tangent_vector(self, point):
        tangent_vector = self.random_point()
        return self.backend.sym(
            tangent_vector / self.norm(point, tangent_vector)
        )


class SkewSymmetric(_Euclidean):
    """(Product) manifold of skew-symmetric matrices.

    Args:
        n: Number of rows and columns of matrices.
        k: Number of elements in the product manifold.

    Note:
        Manifold of ``n x n`` skew-symmetric matrices as a Riemannian
        submanifold of Euclidean space.
        If ``k > 1`` then this is the product manifold of ``k`` skew-symmetric
        ``n x n`` matrices represented as arrays of shape ``(k, n, n)``.
    """

    def __init__(self, n, k=1, backend: Backend = DummyBackendSingleton):
        if k == 1:
            shape = (n, n)
            name = f"Manifold of {n}x{n} skew-symmetric matrices"
        elif k > 1:
            shape = (k, n, n)
            name = f"Product manifold of {k} {n}x{n} skew-symmetric matrices"
        else:
            raise ValueError("k must be an integer no less than 1")
        dimension = int(k * n * (n - 1) / 2)
        super().__init__(name, dimension, *shape, backend=backend)

    def projection(self, point, vector):
        return self.backend.skew(vector)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return self.backend.skew(euclidean_hessian)

    def random_point(self):
        return self.backend.skew(self.backend.random_normal(size=self._shape))

    def random_tangent_vector(self, point):
        tangent_vector = self.random_point()
        return self.backend.skew(
            tangent_vector / self.norm(point, tangent_vector)
        )
