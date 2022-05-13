import numpy as np
from scipy.linalg import expm

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import multiprod, multisym, multitransp


class Stiefel(RiemannianSubmanifold):
    r"""The (product) Stiefel manifold.

    The Stiefel manifold :math:`\St(n, p)` is the manifold of orthonormal ``n x
    p`` matrices.
    A point :math:`\vmX \in \St(n, p)` therefore satisfies the condition
    :math:`\transp{\vmX}\vmX = \Id_p`.
    Points on the manifold are represented as arrays of shape ``(n, p)`` if
    ``k == 1``.
    For ``k > 1``, the class represents the product manifold of ``k`` Stiefel
    manifolds, in which case points on the manifold are represented as arrays
    of shape ``(k, n, p)``.

    The metric is the usual Euclidean metric on :math:`\R^{n \times p}` which
    turns :math:`\St(n, p)^k` into a Riemannian submanifold.

    Args:
        n: The number of rows.
        p: The number of columns.
        k: The number of elements in the product.
        retraction: The type of retraction to use.
            Possible choices are ``qr`` and ``polar``.

    Note:
        The default retraction used here is a first-order one based on
        the QR decomposition.
        To switch to a second-order polar retraction, use ``Stiefel(n, p, k=k,
        retraction="polar")``.
    """

    def __init__(self, n: int, p: int, *, k: int = 1, retraction: str = "qr"):
        self._n = n
        self._p = p
        self._k = k

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )
        if k < 1:
            raise ValueError(f"Need k >= 1. Value supplied was k = {k}")

        if k == 1:
            name = f"Stiefel manifold St({n},{p})"
        elif k >= 2:
            name = f"Product Stiefel manifold St({n},{p})^{k}"
        dimension = int(k * (n * p - p * (p + 1) / 2))
        super().__init__(name, dimension)

        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    @property
    def typical_dist(self):
        return np.sqrt(self._p * self._k)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return vector - multiprod(
            point, multisym(multiprod(multitransp(point), vector))
        )

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        XtG = multiprod(multitransp(point), euclidean_gradient)
        symXtG = multisym(XtG)
        HsymXtG = multiprod(tangent_vector, symXtG)
        return self.projection(point, euclidean_hessian - HsymXtG)

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        if self._k == 1:
            q, r = np.linalg.qr(point + tangent_vector)
            return q @ np.diag(np.sign(np.sign(np.diag(r)) + 0.5))

        target_point = point + tangent_vector
        for i in range(self._k):
            q, r = np.linalg.qr(target_point[i])
            target_point[i] = q @ np.diag(np.sign(np.sign(np.diag(r)) + 0.5))
        return target_point

    def _retraction_polar(self, point, tangent_vector):
        Y = point + tangent_vector
        u, _, vt = np.linalg.svd(Y, full_matrices=False)
        return multiprod(u, vt)

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def random_point(self):
        if self._k == 1:
            matrix = np.random.normal(size=(self._n, self._p))
            q, _ = np.linalg.qr(matrix)
            return q

        point = np.zeros((self._k, self._n, self._p))
        for i in range(self._k):
            point[i], _ = np.linalg.qr(
                np.random.normal(size=(self._n, self._p))
            )
        return point

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape)
        vector = self.projection(point, vector)
        return vector / np.linalg.norm(vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def exp(self, point, tangent_vector):
        if self._k == 1:
            W = expm(
                np.bmat(
                    [
                        [
                            point.T @ tangent_vector,
                            -tangent_vector.T @ tangent_vector,
                        ],
                        [np.eye(self._p), point.T @ tangent_vector],
                    ]
                )
            )
            Z = np.bmat(
                [
                    [expm(-point.T @ tangent_vector)],
                    [np.zeros((self._p, self._p))],
                ]
            )
            return np.bmat([point, tangent_vector]) @ W @ Z

        Y = np.zeros_like(point)
        for i in range(self._k):
            W = expm(
                np.bmat(
                    [
                        [
                            point[i].T @ tangent_vector[i],
                            -tangent_vector[i].T @ tangent_vector[i],
                        ],
                        [np.eye(self._p), point[i].T @ tangent_vector[i]],
                    ]
                )
            )
            Z = np.bmat(
                [
                    [expm(-point[i].T @ tangent_vector[i])],
                    [np.zeros((self._p, self._p))],
                ]
            )
            Y[i] = np.bmat([point[i], tangent_vector[i]]) @ W @ Z
        return Y

    def zero_vector(self, point):
        if self._k == 1:
            return np.zeros((self._n, self._p))
        return np.zeros((self._k, self._n, self._p))
