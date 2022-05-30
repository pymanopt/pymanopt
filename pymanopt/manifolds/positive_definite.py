import numpy as np

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import (
    multiexpm,
    multilogm,
    multiqr,
    multisym,
    multitransp,
)


class SymmetricPositiveDefinite(RiemannianSubmanifold):
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

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def dist(self, point_a, point_b):
        c = np.linalg.cholesky(point_a)
        c_inv = np.linalg.inv(c)
        logm = multilogm(
            c_inv @ point_b @ multitransp(c_inv),
            positive_definite=True,
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

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return point @ multisym(euclidean_gradient) @ point

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return point @ multisym(euclidean_hessian) @ point + multisym(
            tangent_vector @ multisym(euclidean_gradient) @ point
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
            c_inv @ point_b @ multitransp(c_inv),
            positive_definite=True,
        )
        return c @ logm @ multitransp(c)

    def zero_vector(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n))
        return np.zeros((k, n, n))
