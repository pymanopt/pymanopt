from pymanopt.backends import Backend, DummyBackendSingleton
from pymanopt.manifolds.manifold import (
    RiemannianSubmanifold,
    raise_not_implemented_error,
)


class _PositiveDefiniteBase(RiemannianSubmanifold):
    def __init__(
        self,
        name: str,
        n: int,
        k: int,
        dimension: int,
        backend: Backend = DummyBackendSingleton,
    ):
        self._k = k
        self._n = n
        super().__init__(name, dimension, backend=backend)

    @property
    def k(self) -> int:
        return self._k

    @property
    def n(self) -> int:
        return self._n

    @property
    def typical_dist(self):
        return self.backend.sqrt(self.dim)

    def dist(self, point_a, point_b):
        bk = self.backend
        return bk.real(
            bk.linalg_norm(bk.linalg_logm(bk.linalg_solve(point_a, point_b)))
        )

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        bk = self.backend
        p_inv_tv_a = bk.linalg_solve(point, tangent_vector_a)
        p_inv_tv_b = (
            p_inv_tv_a
            if tangent_vector_a is tangent_vector_b
            else bk.linalg_solve(point, tangent_vector_b)
        )
        return bk.real(
            bk.tensordot(bk.conjugate(p_inv_tv_a), p_inv_tv_b, bk.ndim(point))
        )

    def projection(self, point, vector):
        return self.backend.herm(vector)

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, point @ euclidean_gradient @ point)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        bk = self.backend
        return point @ bk.herm(euclidean_hessian) @ point + bk.herm(
            tangent_vector @ bk.herm(euclidean_gradient) @ point
        )

    def norm(self, point, tangent_vector):
        return self.backend.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        bk = self.backend
        # Generate eigenvalues between 1 and 2.
        d = bk.array(
            1.0 + bk.to_real_backend().random_uniform(size=(self.k, self.n, 1))
        )
        # Generate a unitary matrix (with eigenvector columns).
        q, _ = bk.linalg_qr(bk.random_normal(size=(self.n, self.n)))
        # Create a matrix from the eigenvalues and eigenvectors.
        point = q @ (d * bk.conjugate_transpose(q))
        return point if self.k > 1 else point[0]

    def random_tangent_vector(self, point):
        bk = self.backend
        tangent_vector = bk.herm(
            bk.random_randn(self.n, self.n)
            if self.k == 1
            else bk.random_randn(self.k, self.n, self.n)
        )
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def exp(self, point, tangent_vector):
        bk = self.backend
        return point @ bk.linalg_expm(bk.linalg_solve(point, tangent_vector))

    def retraction(self, point, tangent_vector):
        p_inv_tv = self.backend.linalg_solve(point, tangent_vector)
        return self.backend.herm(
            point + tangent_vector + tangent_vector @ p_inv_tv / 2
        )

    def log(self, point_a, point_b):
        bk = self.backend
        return point_a @ bk.linalg_logm(bk.linalg_solve(point_a, point_b))

    def zero_vector(self, point):
        bk = self.backend
        return (
            bk.zeros((self.n, self.n))
            if self.k == 1
            else bk.zeros((self.k, self.n, self.n))
        )


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
        self,
        n: int,
        *,
        k: int = 1,
        backend: Backend = DummyBackendSingleton,
    ):
        if k == 1:
            name = f"Manifold of symmetric positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} "
                f"symmetric positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1) / 2)
        super().__init__(name, n, k, dimension, backend=backend)

    def random_point(self):
        return self.backend.real(super().random_point())


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
        self,
        n: int,
        *,
        k: int = 1,
        backend: Backend = DummyBackendSingleton,
    ):
        if k == 1:
            name = f"Manifold of Hermitian positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} "
                f"Hermitian positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1))
        super().__init__(name, n, k, dimension, backend=backend)


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
        self,
        n: int,
        *,
        k: int = 1,
        backend: Backend = DummyBackendSingleton,
    ):
        if k == 1:
            name = f"Manifold of special Hermitian positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} special "
                f"Hermitian positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1) - k)
        super().__init__(name, n, k, dimension, backend=backend)

    def random_point(self):
        n = self._n
        k = self._k

        # Generate point on the HPD manifold.
        point = super().random_point()

        # Unit determinant.
        shape = (k, 1, 1) if k > 1 else (1, 1)
        det = self.backend.reshape(
            self.backend.linalg_det(point) ** (1 / n), shape
        )
        return point / det

    def random_tangent_vector(self, point):
        tangent_vector = super().random_tangent_vector(point)

        # Project them on tangent space.
        tangent_vector = self.projection(point, tangent_vector)

        # Unit norm.
        return tangent_vector / self.norm(point, tangent_vector)

    def projection(self, point, vector):
        bk = self.backend

        # Project matrix on tangent space of HPD.
        vector = super().projection(point, vector)

        # Project on tangent space of SHPD at x.
        shape = (self.k, 1, 1) if self.k > 1 else (1, 1)
        t = bk.reshape(
            bk.real(bk.trace(bk.linalg_solve(point, vector))), shape
        )
        return vector - (1 / self.n) * t * point

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
        # Compute exponential mapping on HPD.
        e = super().exp(point, tangent_vector)

        # Normalize them. (This is not necessary, but it is good for numerical
        # stability.)
        shape = (self.k, 1, 1) if self.k > 1 else (1, 1)
        det = self.backend.reshape(
            self.backend.linalg_det(e) ** (1 / self.n), shape
        )
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
