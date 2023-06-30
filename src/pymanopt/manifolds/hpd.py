import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from scipy.linalg import sqrtm

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multihconj, multiherm, multilogm, multitransp


class HermitianPositiveDefinite(Manifold):
    """Manifold of (n x n)^k complex Hermitian positive definite matrices."""

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = (
                f"Manifold of Hermitian positive definite ({n} x {n}) matrices"
            )
        else:
            name = f"Product manifold of {k} ({n} x {n}) matrices"
        dimension = 2 * int(k * n * (n + 1) / 2)
        super().__init__(name, dimension)

    def random_point(self):
        # Generate eigenvalues between 1 and 2
        # (eigenvalues of a symmetric matrix are always real).
        d = np.ones((self._k, self._n, 1)) + rnd.rand(self._k, self._n, 1)

        # Generate an orthogonal matrix. Annoyingly qr decomp isn't
        # vectorized so need to use a for loop. Could be done using
        # svd but this is slower for bigger matrices.
        u = np.zeros((self._k, self._n, self._n), dtype=complex)
        for i in range(self._k):
            u[i], _ = la.qr(
                rnd.randn(self._n, self._n) + 1j * rnd.randn(self._n, self._n)
            )

        if self._k == 1:
            return (u @ (d * multihconj(u)))[0]
        return u @ (d * multihconj(u))

    def random_tangent_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            u = multiherm(rnd.randn(n, n) + 1j * rnd.randn(n, n))
        else:
            u = multiherm(rnd.randn(k, n, n) + 1j * rnd.randn(k, n, n))
        return u / self.norm(point, u)

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k != 1:
            return np.zeros((k, n, n), dtype=complex)
        return np.zeros((n, n), dtype=complex)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.real(
            np.tensordot(
                la.solve(point, tangent_vector_a),
                multitransp(la.solve(point, tangent_vector_b)),
                axes=point.ndim,
            )
        )

    def norm(self, point, tangent_vector):
        # This implementation is as fast as np.linalg.solve_triangular and is
        # more stable, as the above solver tends to output non positive
        # definite results.
        c = la.cholesky(point)
        c_inv = la.inv(c)
        return np.real(la.norm(c_inv @ tangent_vector @ multihconj(c_inv)))

    def projection(self, point, vector):
        return multiherm(vector)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return point @ multiherm(euclidean_gradient) @ point

    def exp(self, point, euclidean_gradient):
        k = self._k

        d, q = la.eigh(point)
        if k == 1:
            x_sqrt = q @ np.diag(np.sqrt(d)) @ q.conj().T
            x_isqrt = q @ np.diag(1 / np.sqrt(d)) @ q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_sqrt = q @ temp @ multihconj(q)

            temp = np.zeros(q.shape, dtype=complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(1 / np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_isqrt = q @ temp @ multihconj(q)

        d, q = la.eigh(x_isqrt @ euclidean_gradient @ x_isqrt)
        if k == 1:
            e = q @ np.diag(np.exp(d)) @ q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.exp(d[i, :]))[np.newaxis, :, :]
            d = temp
            e = q @ d @ multihconj(q)

        e = x_sqrt @ e @ x_sqrt
        e = multiherm(e)
        return e

    def retraction(self, point, tangent_vector):
        r = (
            point
            + tangent_vector
            + (1 / 2) * tangent_vector @ la.solve(point, tangent_vector)
        )
        return r

    def log(self, point_a, point_b):
        k = self._k

        d, q = la.eigh(point_a)
        if k == 1:
            x_sqrt = q @ np.diag(np.sqrt(d)) @ q.conj().T
            x_isqrt = q @ np.diag(1 / np.sqrt(d)) @ q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_sqrt = q @ temp @ multihconj(q)

            temp = np.zeros(q.shape, dtype=complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(1 / np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_isqrt = q @ temp @ multihconj(q)

        d, q = la.eigh(x_isqrt @ point_b @ x_isqrt)
        if k == 1:
            log = q @ np.diag(np.log(d)) @ q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.log(d[i, :]))[np.newaxis, :, :]
            d = temp
            log = q @ d @ multihconj(q)

        xi = x_sqrt @ log @ x_sqrt
        xi = multiherm(xi)
        return xi

    def transport(self, point_a, point_b, tangent_vector_a):
        E = multihconj(la.solve(multihconj(point_a), multihconj(point_b)))
        if self._k == 1:
            E = sqrtm(E)
        else:
            for i in range(len(E)):
                E[i, :, :] = sqrtm(E[i, :, :])
        transp_d = E @ tangent_vector_a @ multihconj(E)
        return transp_d

    def dist(self, x, y):
        c = la.cholesky(x)
        c_inv = la.inv(c)
        logm = multilogm(c_inv @ y @ multihconj(c_inv), positive_definite=True)
        return np.real(la.norm(logm))


class SpecialHermitianPositiveDefinite(Manifold):
    """Manifold of (n x n)^k HPD matrices with unit determinant.

    It is a totally geodesic submanifold of the Hermitian positive definite matrices.
    """

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        self.HPD = HermitianPositiveDefinite(n, k)

        if k == 1:
            name = (
                "Manifold of special Hermitian "
                f"positive definite ({n} x {n}) matrices"
            )
            # write same but on two lines
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
        # Generate k HPD matrices.
        x = self.HPD.random_point()

        # Normalize them.
        if self._k == 1:
            x = x / (np.real(la.det(x)) ** (1 / self._n))
        else:
            x = x / (np.real(la.det(x)) ** (1 / self._n)).reshape(-1, 1, 1)

        return x

    def random_tangent_vector(self, point):
        # Generate k matrices.
        k = self._k
        n = self._n
        if k == 1:
            u = rnd.randn(n, n) + 1j * rnd.randn(n, n)
        else:
            u = rnd.randn(k, n, n) + 1j * rnd.randn(k, n, n)

        # Project them on tangent space.
        u = self.projection(point, u)

        # Unit norm.
        u = u / self.norm(point, u)

        return u

    def zero_vector(self, point):
        return self.HPD.zero_vector(point)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self.HPD.inner_product(
            point, tangent_vector_a, tangent_vector_b
        )

    def norm(self, point, tangent_vector):
        return self.HPD.norm(point, tangent_vector)

    def projection(self, point, vector):
        n = self._n
        k = self._k

        # Project matrix on tangent space of HPD.
        u = multiherm(vector)

        # Project on tangent space of SHPD at x.
        t = np.trace(la.solve(point, vector), axis1=-2, axis2=-1)
        if k == 1:
            u = u - (1 / n) * np.real(t) * point
        else:
            u = u - (1 / n) * np.real(t.reshape(-1, 1, 1)) * point

        return u

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        rgrad = point @ euclidean_gradient @ point
        rgrad = self.projection(point, rgrad)
        return rgrad

    def exp(self, point, tangent_vector):
        e = self.HPD.exp(point, tangent_vector)

        # Normalize them.
        if self._k == 1:
            e = e / np.real(la.det(e)) ** (1 / self._n)
        else:
            e = e / (np.real(la.det(e)) ** (1 / self._n)).reshape(-1, 1, 1)
        return e

    def retraction(self, point, tangent_vector):
        r = self.HPD.retraction(point, tangent_vector)

        # Normalize them.
        if self._k == 1:
            r = r / np.real(la.det(r)) ** (1 / self._n)
        else:
            r = r / (np.real(la.det(r)) ** (1 / self._n)).reshape(-1, 1, 1)
        return r

    def log(self, point_a, point_b):
        return self.HPD.log(point_a, point_b)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(
            point_b, self.HPD.transport(point_a, point_b, tangent_vector_a)
        )

    def dist(self, point_a, point_b):
        return self.HPD.dist(point_a, point_b)
