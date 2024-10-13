from typing import Optional

from pymanopt.manifolds.manifold import (
    RiemannianSubmanifold,
    raise_not_implemented_error,
)
from pymanopt.numerics import NumericsBackend


class _PositiveDefiniteBase(RiemannianSubmanifold):
    def __init__(
        self,
        name,
        dimension,
        *shape,
        backend: Optional[NumericsBackend] = None,
    ):
        self._shape = shape
        super().__init__(name, dimension, backend=backend)

    @property
    def typical_dist(self):
        return self.backend.sqrt(self.dim)

    def dist(self, point_a, point_b):
        c = self.backend.linalg_cholesky(point_a)
        c_inv = self.backend.linalg_inv(c)
        logm = self.backend.linalg_logm(
            c_inv @ point_b @ self.backend.conjugate_transpose(c_inv),
            positive_definite=True,
        )
        return self.backend.real(self.backend.linalg_norm(logm))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        p_inv_tv_a = self.backend.linalg_solve(point, tangent_vector_a)
        if tangent_vector_a is tangent_vector_b:
            p_inv_tv_b = p_inv_tv_a
        else:
            p_inv_tv_b = self.backend.linalg_solve(point, tangent_vector_b)
        return self.backend.real(
            self.backend.tensordot(
                p_inv_tv_a,
                self.backend.transpose(p_inv_tv_b),
                axes=point.ndim,
            )
        )

    def projection(self, point, vector):
        return self.backend.herm(vector)

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, point @ euclidean_gradient @ point)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return point @ self.backend.herm(
            euclidean_hessian
        ) @ point + self.backend.herm(
            tangent_vector @ self.backend.herm(euclidean_gradient) @ point
        )

    def norm(self, point, tangent_vector):
        return self.backend.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        # Generate eigenvalues between 1 and 2.
        d = 1.0 + self.backend.random_uniform(size=(self._k, self._n, 1))

        # Generate a unitary matrix.
        q, _ = self.backend.linalg_qr(
            self.backend.random_normal(size=(self._n, self._n))
            + 1j * self.backend.random_normal(size=(self._n, self._n))
        )
        point = q @ (d * self.backend.conjugate_transpose(q))
        return point if self._k > 1 else point[0]

    def random_tangent_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            tangent_vector = self.backend.random_randn(n, n)
            if self.backend.iscomplexobj(point):
                tangent_vector = (
                    tangent_vector + 1j * self.backend.random_randn(n, n)
                )
        else:
            tangent_vector = self.backend.random_randn(k, n, n)
            if self.backend.iscomplexobj(point):
                tangent_vector = (
                    tangent_vector + 1j * self.backend.random_randn(k, n, n)
                )
        tangent_vector = self.backend.herm(tangent_vector)
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def exp(self, point, tangent_vector):
        p_inv_tv = self.backend.linalg_solve(point, tangent_vector)
        return point @ self.backend.linalg_expm(p_inv_tv, symmetric=False)

    def retraction(self, point, tangent_vector):
        p_inv_tv = self.backend.linalg_solve(point, tangent_vector)
        return self.backend.herm(
            point + tangent_vector + tangent_vector @ p_inv_tv / 2
        )

    def log(self, point_a, point_b):
        c = self.backend.linalg_cholesky(point_a)
        c_inv = self.backend.linalg_inv(c)
        logm = self.backend.linalg_logm(
            c_inv @ point_b @ self.backend.conjugate_transpose(c_inv),
            positive_definite=True,
        )
        return c @ logm @ self.backend.conjugate_transpose(c)

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return self.backend.zeros((n, n), dtype=point.dtype)
        return self.backend.zeros((k, n, n), dtype=point.dtype)


class SymmetricPositiveDefinite(_PositiveDefiniteBase):
    """Manifold of symmetric positive definite matrices.

    Points on the manifold and tangent vectors are represented as arrays of
    shape ``k x n x n`` if ``k > 1``, and ``n x n`` if ``k == 1``.

    Args:
        n: The size of matrices in the manifold, i.e., the number of rows and
           columns of each element.
        k: The number of elements in the product geometry.

    Note:
        The geometry is based on the discussion in chapter 6 of [Bha2007]_.
        Also see [SH2015]_ for more details.

        The second-order retraction is taken from [JVV2012]_.
    """

    def __init__(
        self, n: int, *, k: int = 1, backend: Optional[NumericsBackend] = None
    ):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of symmetric positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} "
                f"symmetric positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1) / 2)
        super().__init__(name, dimension, backend=backend)

    def random_point(self):
        return super().random_point().real


class HermitianPositiveDefinite(_PositiveDefiniteBase):
    """Manifold of Hermitian positive definite matrices.

    Points on the manifold and tangent vectors are represented as arrays of
    shape ``k x n x n`` if ``k > 1``, and ``n x n`` if ``k == 1``.

    Args:
        n: The size of matrices in the manifold, i.e., the number of rows and
           columns of each element.
        k: The number of elements in the product geometry.
    """

    IS_COMPLEX = True

    def __init__(
        self, n: int, *, k: int = 1, backend: Optional[NumericsBackend] = None
    ):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of Hermitian positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} "
                f"Hermitian positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1))
        super().__init__(name, dimension, backend=backend)


class SpecialHermitianPositiveDefinite(_PositiveDefiniteBase):
    """Manifold of hermitian positive definite matrices with unit determinant.

    Points on the manifold and tangent vectors are represented as arrays of
    shape ``k x n x n`` if ``k > 1``, and ``n x n`` if ``k == 1``.

    Args:
        n: The size of matrices in the manifold, i.e., the number of rows and
           columns of each element.
        k: The number of elements in the product geometry.
    """

    IS_COMPLEX = True

    def __init__(
        self, n: int, *, k: int = 1, backend: Optional[NumericsBackend] = None
    ):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of special Hermitian positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} special "
                f"Hermitian positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1) - k)
        super().__init__(name, dimension, backend=backend)

    def random_point(self):
        n = self._n
        k = self._k

        # Generate point on the HPD manifold.
        point = super().random_point()

        # Unit determinant.
        shape = (k, 1, 1) if k > 1 else (1, 1)
        det = (self.backend.linalg_det(point) ** (1 / n)).reshape(shape)
        return point / det

    def random_tangent_vector(self, point):
        tangent_vector = super().random_tangent_vector(point)

        # Project them on tangent space.
        tangent_vector = self.projection(point, tangent_vector)

        # Unit norm.
        return tangent_vector / self.norm(point, tangent_vector)

    def projection(self, point, vector):
        n = self._n
        k = self._k

        # Project matrix on tangent space of HPD.
        vector = super().projection(point, vector)

        # Project on tangent space of SHPD at x.
        shape = (k, 1, 1) if k > 1 else (1, 1)
        t = self.backend.real(
            self.backend.trace(
                self.backend.linalg_solve(point, vector), axis1=-2, axis2=-1
            )
        ).reshape(shape)
        return vector - (1 / n) * t * point

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(
            point,
            super().euclidean_to_riemannian_gradient(
                point, euclidean_gradient
            ),
        )

    @raise_not_implemented_error
    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        pass

    def exp(self, point, tangent_vector):
        n = self._n
        k = self._k

        # Compute exponential mapping on HPD.
        e = super().exp(point, tangent_vector)

        # Normalize them. (This is not necessary, but it is good for numerical
        # stability.)
        shape = (k, 1, 1) if k > 1 else (1, 1)
        det = (self.backend.linalg_det(e) ** (1 / n)).reshape(shape)
        return e / det

    def retraction(self, point, tangent_vector):
        n = self._n
        k = self._k

        # Compute retraction on HPD.
        r = super().retraction(point, tangent_vector)

        # Unit determinant.
        shape = (k, 1, 1) if k > 1 else (1, 1)
        det = (self.backend.linalg_det(r) ** (1 / n)).reshape(shape)
        return r / det

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(
            point_b, super().projection(point_b, tangent_vector_a)
        )
