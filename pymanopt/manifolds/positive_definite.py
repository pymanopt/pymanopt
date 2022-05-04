import numpy as np
from scipy.linalg import expm

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp


class SymmetricPositiveDefinite(RiemannianSubmanifold):
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
    def typical_dist(self):
        return np.sqrt(self.dim)

    def dist(self, point_a, point_b):
        # Adapted from equation (6.13) of [Bha2007].
        c = np.linalg.cholesky(point_a)
        c_inv = np.linalg.inv(c)
        logm = multilog(
            multiprod(multiprod(c_inv, point_b), multitransp(c_inv)),
            pos_def=True,
        )
        return np.linalg.norm(logm)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        p_inv_tv_a = np.linalg.solve(point, tangent_vector_a)
        if tangent_vector_a is tangent_vector_b:
            p_inv_tv_b = p_inv_tv_a
        else:
            p_inv_tv_b = np.linalg.solve(point, tangent_vector_b)
        return np.tensordot(
            p_inv_tv_a, multitransp(p_inv_tv_b), axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return multisym(vector)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        # TODO: Check that this is correct
        return multiprod(multiprod(point, multisym(euclidean_gradient)), point)

    def euclidean_to_riemannian_hvp(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        # TODO: Check that this is correct
        return multiprod(
            multiprod(point, multisym(euclidean_hvp)), point
        ) + multisym(
            multiprod(
                multiprod(tangent_vector, multisym(euclidean_gradient)), point
            )
        )

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        # Generate eigenvalues between 1 and 2.
        d = np.ones((self._k, self._n, 1)) + np.random.uniform(
            size=(self._k, self._n, 1)
        )

        # Generate an orthogonal matrix.
        u = np.zeros((self._k, self._n, self._n))
        for i in range(self._k):
            u[i], _ = np.linalg.qr(np.random.normal(size=(self._n, self._n)))

        if self._k == 1:
            return multiprod(u, d * multitransp(u))[0]
        return multiprod(u, d * multitransp(u))

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
        if self._k > 1:
            e = np.zeros(np.shape(point))
            for i in range(self._k):
                e[i] = expm(p_inv_tv[i])
        else:
            e = expm(p_inv_tv)
        return multiprod(point, e)

    retraction = exp

    def log(self, point_a, point_b):
        c = np.linalg.cholesky(point_a)
        c_inv = np.linalg.inv(c)
        logm = multilog(
            multiprod(multiprod(c_inv, point_b), multitransp(c_inv)),
            pos_def=True,
        )
        return multiprod(multiprod(c, logm), multitransp(c))

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n))
        return np.zeros((k, n, n))
