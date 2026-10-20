import numpy as np
import scipy

from pymanopt.manifolds.manifold import Manifold


class GeneralizedStiefel(Manifold):
    r"""The Generalized Stiefel manifold.

    The Generalized Stiefel manifold :math:`\St(n, p, B)` is the
    manifold of orthonormal ``n x p`` matrices w.r.t. a symmetric
    positive definite matrix :math:`\vmB\in\R^{n\times n}`.
    A point :math:`\vmX \in \St(n, p, B)` therefore satisfies the condition
    :math:`\transp{\vmX}\vmB\vmX = \Id_p`.
    Points on the manifold are represented as arrays of shape ``(n, p)``.

    Args:
        n: The number of rows.
        p: The number of columns.
        B: A symmetric positive definite matrix
        Binv: Inverse of B if known
        retraction: The type of retraction to use.
            Possible choices are ``qr`` and ``polar``.

    Note:
        The matrix :math:`\vmB` can be provided as a numpy array, a
        scipy sparse matrix, or a scipy linear operator.

        The default retraction used here is a first-order one based on
        the QR decomposition.
        To switch to a second-order polar retraction, use
        ``GeneralizedStiefel(n, p, B, retraction="polar")``.

        Obtaining the Riemannian gradient from the Euclidean gradient requires
        the inverse of :math:`\vmB`. If this is known, the inverse can be provided
        to speed up this operation. Otherwise, the LU transorm is computed and
        stored the first time this function is called. If :math:`\vmB` is given
        as a scipy linear operator, :math:`\vmB^{-1}` must be supplied.
    """

    def __init__(
        self,
        n: int,
        p: int,
        B: np.array,
        *,
        Binv: np.array = None,
        retraction: str = "qr",
    ):
        self._n = n
        self._p = p
        self.B = B
        self.Binv = Binv
        self.LU = None
        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )
        name = f"Generalized Stiefel manifold St({n},{p},B)"
        dimension = int((n * p - p * (p + 1) / 2))
        super().__init__(name, dimension)
        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    @property
    def typical_dist(self):
        return np.sqrt(self._p)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.vdot(tangent_vector_a, self.B @ tangent_vector_b)

    def projection(self, point, vector):
        return vector - point @ self.sym(point.T @ self.B @ vector)

    def sym(self, A):
        """Returns the symmetric part of A."""
        return 0.5 * (A + A.T)

    to_tangent_space = projection

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        Y = point + tangent_vector
        return self.gqf(Y)

    def _retraction_polar(self, point, tangent_vector):
        Y = point + tangent_vector
        return self.guf(Y)

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        if self._retraction == self._retraction_qr:
            point = self.gqf(np.random.normal(size=(self._n, self._p)))
        elif self._retraction == self._retraction_polar:
            point = self.guf(np.random.normal(size=(self._n, self._p)))
        return point

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape)
        vector = self.projection(point, vector)
        return vector / np.linalg.norm(vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        if self.Binv is not None:
            egrad_scaled = self.Binv @ euclidean_gradient
        else:
            if self.LU is None:
                self.LU = scipy.sparse.linalg.splu(self.B)
                egrad_scaled = self.LU.solve(euclidean_gradient)
            else:
                egrad_scaled = self.LU.solve(euclidean_gradient)
        rgrad = egrad_scaled - point @ self.sym(point.T @ euclidean_gradient)
        return rgrad

    def zero_vector(self, point):
        return np.zeros((self._n, self._p))

    def guf(self, Y):
        """Generalized polar decomposition."""
        U, _, Vh = np.linalg.svd(Y, full_matrices=False)
        ssquare, q = np.linalg.eig(U.T @ self.B @ U)
        qsinv = q / np.sqrt(ssquare)
        X = U @ (qsinv @ q.T @ Vh)
        return X.real

    def gqf(self, Y):
        """Generalized QR decomposition.

        See algorithm 3.1 in [SA2019]_
        """
        # Generalized QR decomposition
        # Algorithm 3.1 in https://doi.org/10.1007/s10589-018-0046-7
        R = scipy.linalg.cholesky(self.sym(Y.T @ self.B @ Y))
        # R is upper triangular
        X_T = scipy.linalg.solve_triangular(R.T, Y.T, lower=True)
        return X_T.T
