import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from scipy.linalg import expm

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp


class SymmetricPositiveDefinite(EuclideanEmbeddedSubmanifold):
    """Manifold of symmetric positive definite matrices.

    Notes:
        The geometry is based on the discussion in chapter 6 of [Bha2007]_.
        Also see [SH2015]_ for more details.
    """

    def __init__(self, n, k=1):
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

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def dist(self, x, y):
        # Adapted from equation 6.13 of "Positive definite matrices". The
        # Cholesky decomposition gives the same result as matrix sqrt. There
        # may be more efficient ways to compute this.
        c = la.cholesky(x)
        c_inv = la.inv(c)
        logm = multilog(
            multiprod(multiprod(c_inv, y), multitransp(c_inv)), pos_def=True
        )
        return la.norm(logm)

    def inner(self, x, u, v):
        xinvu = la.solve(x, u)
        if u is v:
            xinvv = xinvu
        else:
            xinvv = la.solve(x, v)
        return np.tensordot(xinvu, multitransp(xinvv), axes=x.ndim)

    def proj(self, X, G):
        return multisym(G)

    def egrad2rgrad(self, x, u):
        # TODO: Check that this is correct
        return multiprod(multiprod(x, multisym(u)), x)

    def ehess2rhess(self, x, egrad, ehess, u):
        # TODO: Check that this is correct
        return multiprod(multiprod(x, multisym(ehess)), x) + multisym(
            multiprod(multiprod(u, multisym(egrad)), x)
        )

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def rand(self):
        # The way this is done is arbitrary. I think the space of p.d.
        # matrices would have infinite measure w.r.t. the Riemannian metric
        # (cf. integral 0-inf [ln(x)] dx = inf) so impossible to have a
        # 'uniform' distribution.

        # Generate eigenvalues between 1 and 2
        d = np.ones((self._k, self._n, 1)) + rnd.rand(self._k, self._n, 1)

        # Generate an orthogonal matrix. Annoyingly qr decomp isn't
        # vectorized so need to use a for loop. Could be done using
        # svd but this is slower for bigger matrices.
        u = np.zeros((self._k, self._n, self._n))
        for i in range(self._k):
            u[i], r = la.qr(rnd.randn(self._n, self._n))

        if self._k == 1:
            return multiprod(u, d * multitransp(u))[0]
        return multiprod(u, d * multitransp(u))

    def randvec(self, x):
        k = self._k
        n = self._n
        if k == 1:
            u = multisym(rnd.randn(n, n))
        else:
            u = multisym(rnd.randn(k, n, n))
        return u / self.norm(x, u)

    def transp(self, x1, x2, d):
        return d

    def exp(self, x, u):
        # TODO: Check which method is faster depending on n, k.
        x_inv_u = la.solve(x, u)
        if self._k > 1:
            e = np.zeros(np.shape(x))
            for i in range(self._k):
                e[i] = expm(x_inv_u[i])
        else:
            e = expm(x_inv_u)
        return multiprod(x, e)
        # This alternative implementation is sometimes faster though less
        # stable. It can return a matrix with small negative determinant.
        #    c = la.cholesky(x)
        #    c_inv = la.inv(c)
        #    e = multiexp(multiprod(multiprod(c_inv, u), multitransp(c_inv)),
        #                 sym=True)
        #    return multiprod(multiprod(c, e), multitransp(c))

    retr = exp

    def log(self, x, y):
        c = la.cholesky(x)
        c_inv = la.inv(c)
        logm = multilog(
            multiprod(multiprod(c_inv, y), multitransp(c_inv)), pos_def=True
        )
        return multiprod(multiprod(c, logm), multitransp(c))

    def zerovec(self, x):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n))
        return np.zeros((k, n, n))
