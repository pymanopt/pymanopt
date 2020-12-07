import numpy as np
from numpy import linalg as la, random as rnd
from scipy.linalg import sqrtm

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multihconj, multiherm, multilog
from pymanopt.tools.multi import multiprod, multitransp


class HermitianPositiveDefinite(EuclideanEmbeddedSubmanifold):
    """Manifold of (n x n)^k complex Hermitian positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = ("Manifold of Hermitian positive definite\
                    ({} x {}) matrices").format(n, n)
        else:
            name = "Product manifold of {} ({} x {}) matrices".format(k, n, n)
        dimension = 2 * int(k * n * (n + 1) / 2)
        super().__init__(name, dimension)

    def rand(self):
        # Generate eigenvalues between 1 and 2
        # (eigenvalues of a symmetric matrix are always real).
        d = np.ones((self._k, self._n, 1)) + rnd.rand(self._k, self._n, 1)

        # Generate an orthogonal matrix. Annoyingly qr decomp isn't
        # vectorized so need to use a for loop. Could be done using
        # svd but this is slower for bigger matrices.
        u = np.zeros((self._k, self._n, self._n), dtype=np.complex)
        for i in range(self._k):
            u[i], r = la.qr(
                rnd.randn(self._n, self._n)+1j*rnd.randn(self._n, self._n))

        if self._k == 1:
            return multiprod(u, d * multihconj(u))[0]
        return multiprod(u, d * multihconj(u))

    def randvec(self, x):
        k = self._k
        n = self._n
        if k == 1:
            u = multiherm(rnd.randn(n, n)+1j*rnd.randn(n, n))
        else:
            u = multiherm(rnd.randn(k, n, n)+1j*rnd.randn(k, n, n))
        return u / self.norm(x, u)

    def zerovec(self, x):
        k = self._k
        n = self._n
        if k != 1:
            return np.zeros((k, n, n), dtype=np.complex)
        return np.zeros((n, n), dtype=np.complex)

    def inner(self, x, u, v):
        return np.real(
            np.tensordot(la.solve(x, u), multitransp(la.solve(x, v)),
                         axes=x.ndim))

    def norm(self, x, u):
        # This implementation is as fast as np.linalg.solve_triangular and is
        # more stable, as the above solver tends to output non positive
        # definite results.
        c = la.cholesky(x)
        c_inv = la.inv(c)
        return np.real(
            la.norm(multiprod(multiprod(c_inv, u), multihconj(c_inv))))

    def proj(self, X, G):
        return multiherm(G)

    def egrad2rgrad(self, x, u):
        return multiprod(multiprod(x, multiherm(u)), x)

    def exp(self, x, u):
        k = self._k

        d, q = la.eigh(x)
        if k == 1:
            x_sqrt = q@np.diag(np.sqrt(d))@q.conj().T
            x_isqrt = q@np.diag(1/np.sqrt(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_sqrt = multiprod(multiprod(q, temp), multihconj(q))

            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(1/np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_isqrt = multiprod(multiprod(q, temp), multihconj(q))

        d, q = la.eigh(multiprod(multiprod(x_isqrt, u), x_isqrt))
        if k == 1:
            e = q@np.diag(np.exp(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.exp(d[i, :]))[np.newaxis, :, :]
            d = temp
            e = multiprod(multiprod(q, d), multihconj(q))

        e = multiprod(multiprod(x_sqrt, e), x_sqrt)
        e = multiherm(e)
        return e

    def retr(self, x, u):
        r = x + u + (1/2)*u@la.solve(x, u)
        return r

    def log(self, x, y):
        k = self._k

        d, q = la.eigh(x)
        if k == 1:
            x_sqrt = q@np.diag(np.sqrt(d))@q.conj().T
            x_isqrt = q@np.diag(1/np.sqrt(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_sqrt = multiprod(multiprod(q, temp), multihconj(q))

            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(1/np.sqrt(d[i, :]))[np.newaxis, :, :]
            x_isqrt = multiprod(multiprod(q, temp), multihconj(q))

        d, q = la.eigh(multiprod(multiprod(x_isqrt, y), x_isqrt))
        if k == 1:
            log = q@np.diag(np.log(d))@q.conj().T
        else:
            temp = np.zeros(q.shape, dtype=np.complex)
            for i in range(q.shape[0]):
                temp[i, :, :] = np.diag(np.log(d[i, :]))[np.newaxis, :, :]
            d = temp
            log = multiprod(multiprod(q, d), multihconj(q))

        xi = multiprod(multiprod(x_sqrt, log), x_sqrt)
        xi = multiherm(xi)
        return xi

    def transp(self, x1, x2, d):
        E = multihconj(la.solve(multihconj(x1), multihconj(x2)))
        if self._k == 1:
            E = sqrtm(E)
        else:
            for i in range(len(E)):
                E[i, :, :] = sqrtm(E[i, :, :])
        transp_d = multiprod(multiprod(E, d), multihconj(E))
        return transp_d

    def dist(self, x, y):
        c = la.cholesky(x)
        c_inv = la.inv(c)
        logm = multilog(multiprod(multiprod(c_inv, y), multihconj(c_inv)),
                        pos_def=True)
        return np.real(la.norm(logm))


class SpecialHermitianPositiveDefinite(EuclideanEmbeddedSubmanifold):
    """Manifold of (n x n)^k Hermitian positive
    definite matrices with unit determinant
    called 'Special Hermitian positive definite manifold'.
    It is a totally geodesic submanifold of
    the Hermitian positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        self.HPD = HermitianPositiveDefinite(n, k)

        if k == 1:
            name = ("Manifold of special Hermitian positive definite\
                    ({} x {}) matrices").format(n, n)
        else:
            name = "Product manifold of {} special Hermitian positive\
                    definite ({} x {}) matrices".format(k, n, n)
        dimension = int(k * (n*(n+1) - 1))
        super().__init__(name, dimension)

    def rand(self):
        # Generate k HPD matrices.
        x = self.HPD.rand()

        # Normalize them.
        if self._k == 1:
            x = x / (np.real(la.det(x))**(1/self._n))
        else:
            x = x / (np.real(la.det(x))**(1/self._n)).reshape(-1, 1, 1)

        return x

    def randvec(self, x):
        # Generate k matrices.
        k = self._k
        n = self._n
        if k == 1:
            u = rnd.randn(n, n)+1j*rnd.randn(n, n)
        else:
            u = rnd.randn(k, n, n)+1j*rnd.randn(k, n, n)

        # Project them on tangent space.
        u = self.proj(x, u)

        # Unit norm.
        u = u / self.norm(x, u)

        return u

    def zerovec(self, x):
        return self.HPD.zerovec(x)

    def inner(self, x, u, v):
        return self.HPD.inner(x, u, v)

    def norm(self, x, u):
        return self.HPD.norm(x, u)

    def proj(self, x, u):
        n = self._n
        k = self._k

        # Project matrix on tangent space of HPD.
        u = multiherm(u)

        # Project on tangent space of SHPD at x.
        t = np.trace(la.solve(x, u), axis1=-2, axis2=-1)
        if k == 1:
            u = u - (1/n) * np.real(t) * x
        else:
            u = u - (1/n) * np.real(t.reshape(-1, 1, 1)) * x

        return u

    def egrad2rgrad(self, x, u):
        rgrad = multiprod(multiprod(x, u), x)
        rgrad = self.proj(x, rgrad)
        return rgrad

    def exp(self, x, u):
        e = self.HPD.exp(x, u)

        # Normalize them.
        if self._k == 1:
            e = e / np.real(la.det(e))**(1/self._n)
        else:
            e = e / (np.real(la.det(e))**(1/self._n)).reshape(-1, 1, 1)
        return e

    def retr(self, x, u):
        r = self.HPD.retr(x, u)

        # Normalize them.
        if self._k == 1:
            r = r / np.real(la.det(r))**(1/self._n)
        else:
            r = r / (np.real(la.det(r))**(1/self._n)).reshape(-1, 1, 1)
        return r

    def log(self, x, y):
        return self.HPD.log(x, y)

    def transp(self, x1, x2, d):
        return self.proj(x2, self.HPD.transp(x1, x2, d))

    def dist(self, x, y):
        return self.HPD.dist(x, y)
