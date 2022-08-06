import numpy as np

from pymanopt.manifolds.manifold import Manifold


class Positive(Manifold):
    r"""The (product) manifold of positive matrices.

    Args:
        m: The number of rows.
        n: The number of columns.
        k: The number of matrices in the product.
        use_parallel_transport: Flag whether to use a parallel transport for
            :meth:`transport` or a transporter (the default).

    Note:
        Points on the manifold are represented as arrays of size ``m x n``
        (when ``k`` is 1), and ``k x m x n`` otherwise.

        The tangent spaces of the manifold correspond to copies of :math:`\R^{m
        \times n}`.
        As such, tangent vectors are represented as arrays of the same shape as
        points on the manifold without any positivity constraints on the
        individual elements.

        The Riemannian metric is the bi-invariant metric for positive definite
        matrices from chapter 6 of [Bha2007]_ on individual scalar coordinates
        of matrices.
        See also section 11.4 of [Bou2020]_ for further details.

        The second-order retraction is taken from [JVV2012]_.

        The parallel transport that is used when ``use_parallel_transport`` is
        ``True`` is taken from [SH2015]_.
    """

    def __init__(
        self,
        m: int,
        n: int,
        *,
        k: int = 1,
        use_parallel_transport: bool = False,
    ):
        self._m = m
        self._n = n
        self._k = k

        if use_parallel_transport:
            self._transport = self._parallel_transport
        else:
            self._transport = self._transporter

        if k == 1:
            name = f"Manifold of positive {m}x{n} matrices"
        else:
            name = f"Product manifold of {k} positive {m}x{n} matrices"
        dimension = int(k * m * n)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a / point,
            tangent_vector_b / point,
            axes=tangent_vector_a.ndim,
        )

    def projection(self, point, vector):
        return vector

    to_tangent_space = projection

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        point = np.exp(np.random.normal(size=(self._k, self._m, self._n)))
        if self._k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape) * point
        return vector / self.norm(point, vector)

    def zero_vector(self, point):
        return np.zeros(point.shape)

    def dist(self, point_a, point_b):
        log_ratio = np.log(point_a) - np.log(point_b)
        return np.sqrt(np.tensordot(log_ratio, log_ratio, axes=point_a.ndim))

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return euclidean_gradient * point**2

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return (
            euclidean_hessian * point**2
            + tangent_vector * euclidean_gradient * point
        )

    def exp(self, point, tangent_vector):
        return point * np.exp(tangent_vector / point)

    def log(self, point_a, point_b):
        return point_a * (np.log(point_b) - np.log(point_a))

    def retraction(self, point, tangent_vector):
        return point + tangent_vector + tangent_vector**2 / point / 2

    def _transporter(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def _parallel_transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a * point_b / point_a

    def transport(self, point_a, point_b, tangent_vector_a):
        return self._transport(point_a, point_b, tangent_vector_a)
