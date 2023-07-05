import numpy as np

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import (
    multiexpm,
    multihconj,
    multiherm,
    multilogm,
    multiqr,
    multisym,
    multitransp,
)


class _positive_definite(RiemannianSubmanifold):
    def __init__(self, name, dimension, *shape):
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def dist(self, point_a, point_b):
        c = np.linalg.cholesky(point_a)
        c_inv = np.linalg.inv(c)
        logm = multilogm(
            c_inv @ point_b @ multihconj(c_inv),
            positive_definite=True,
        )
        return np.real(np.linalg.norm(logm))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.real(
            np.tensordot(
                np.linalg.solve(point, tangent_vector_a),
                multitransp(np.linalg.solve(point, tangent_vector_b)),
                axes=point.ndim,
            )
        )

    def projection(self, point, vector):
        return multiherm(vector)

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, point @ euclidean_gradient @ point)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return point @ multiherm(euclidean_hessian) @ point + multiherm(
            tangent_vector @ multiherm(euclidean_gradient) @ point
        )

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        # Generate eigenvalues between 1 and 2.
        d = 1.0 + np.random.uniform(size=(self._k, self._n, 1))

        # Generate an orthogonal matrix.
        q, _ = multiqr(np.random.normal(size=(self._n, self._n)))
        point = q @ (d * multitransp(q))
        if self._k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            tangent_vector = multisym(np.random.normal(size=(n, n)))
        else:
            tangent_vector = multisym(np.random.normal(size=(k, n, n)))
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def exp(self, point, tangent_vector):
        p_inv_tv = np.linalg.solve(point, tangent_vector)
        return point @ multiexpm(p_inv_tv, symmetric=False)

    def retraction(self, point, tangent_vector):
        p_inv_tv = np.linalg.solve(point, tangent_vector)
        return multisym(point + tangent_vector + tangent_vector @ p_inv_tv / 2)

    def log(self, point_a, point_b):
        c = np.linalg.cholesky(point_a)
        c_inv = np.linalg.inv(c)
        logm = multilogm(
            c_inv @ point_b @ multihconj(c_inv),
            positive_definite=True,
        )
        return c @ logm @ multihconj(c)

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n))
        return np.zeros((k, n, n))


class SymmetricPositiveDefinite(_positive_definite):
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

    def __init__(self, n: int, *, k: int = 1):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1) / 2)
        super().__init__(name, dimension)


class HermitianPositiveDefinite(_positive_definite):
    """Manifold of hermitian positive definite matrices.

    Points on the manifold and tangent vectors are represented as arrays of
    shape ``k x n x n`` if ``k > 1``, and ``n x n`` if ``k == 1``.

    Args:
        n: The size of matrices in the manifold, i.e., the number of rows and
            columns of each element.
        k: The number of elements in the product geometry.
    """

    def __init__(self, n: int, *, k: int = 1):
        self._n = n
        self._k = k

        if k == 1:
            name = (
                f"Manifold of Hermitian positive definite ({n} x {n}) matrices"
            )
        else:
            name = f"Product manifold of {k} ({n} x {n}) Hermitian positive definite"
        dimension = 2 * int(k * n * (n + 1) / 2)
        super().__init__(name, dimension)

    def random_point(self):
        # Generate eigenvalues between 1 and 2.
        d = 1.0 + np.random.uniform(size=(self._k, self._n, 1))

        # Generate an orthogonal matrix.
        q, _ = multiqr(
            np.random.normal(size=(self._n, self._n))
            + 1j * np.random.normal(size=(self._n, self._n))
        )
        point = q @ (d * multihconj(q))
        if self._k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            tangent_vector = multiherm(
                np.random.randn(n, n) + 1j * np.random.randn(n, n)
            )
        else:
            tangent_vector = multiherm(
                np.random.randn(k, n, n) + 1j * np.random.randn(k, n, n)
            )
        return tangent_vector / self.norm(point, tangent_vector)

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n), dtype=complex)
        return np.zeros((k, n, n), dtype=complex)


class SpecialHermitianPositiveDefinite(_positive_definite):
    """Manifold of hermitian positive definite matrices with unit determinant.

    Points on the manifold and tangent vectors are represented as arrays of
    shape ``k x n x n`` if ``k > 1``, and ``n x n`` if ``k == 1``.

    Args:
        n: The size of matrices in the manifold, i.e., the number of rows and
            columns of each element.
        k: The number of elements in the product geometry.
    """

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        self.HPD = HermitianPositiveDefinite(n, k=k)

        if k == 1:
            name = (
                "Manifold of special Hermitian "
                f"positive definite ({n} x {n}) matrices"
            )
        else:
            name = (
                f"Product manifold of {k} special "
                "Hermitian positive definite "
                f"({n} x {n}) matrices"
            )
        dimension = int(k * (n * (n + 1) - 1))
        super().__init__(name, dimension)

    def random_point(self):
        # Generate eigenvalues between 1 and 2.
        d = 1.0 + np.random.uniform(size=(self._k, self._n, 1))

        # Generate an orthogonal matrix.
        q, _ = multiqr(
            np.random.normal(size=(self._n, self._n))
            + 1j * np.random.normal(size=(self._n, self._n))
        )
        point = q @ (d * multihconj(q))

        point = point / (
            np.real(np.linalg.det(point)) ** (1 / self._n)
        ).reshape(-1, 1, 1)

        return point

    def random_tangent_vector(self, point):
        # Generate k matrices.
        k = self._k
        n = self._n
        if k == 1:
            u = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        else:
            u = np.random.randn(k, n, n) + 1j * np.random.randn(k, n, n)

        # Project them on tangent space.
        u = self.projection(point, u)

        # Unit norm.
        u = u / self.norm(point, u)

        return u

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n), dtype=complex)
        return np.zeros((k, n, n), dtype=complex)

    def projection(self, point, vector):
        n = self._n
        k = self._k

        # Project matrix on tangent space of HPD.
        u = multiherm(vector)

        # Project on tangent space of SHPD at x.
        t = np.trace(np.linalg.solve(point, vector), axis1=-2, axis2=-1)
        if k == 1:
            u = u - (1 / n) * np.real(t) * point
        else:
            u = u - (1 / n) * np.real(t.reshape(-1, 1, 1)) * point

        return u

    def exp(self, point, tangent_vector):
        e = super().exp(point, tangent_vector)

        # Normalize them.
        if self._k == 1:
            e = e / np.real(np.linalg.det(e)) ** (1 / self._n)
        else:
            e = e / (np.real(np.linalg.det(e)) ** (1 / self._n)).reshape(
                -1, 1, 1
            )
        return e

    def retraction(self, point, tangent_vector):
        r = super().retraction(point, tangent_vector)

        # Normalize them.
        if self._k == 1:
            r = r / np.real(np.linalg.det(r)) ** (1 / self._n)
        else:
            r = r / (np.real(np.linalg.det(r)) ** (1 / self._n)).reshape(
                -1, 1, 1
            )
        return r

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(
            point_b, self.HPD.transport(point_a, point_b, tangent_vector_a)
        )
