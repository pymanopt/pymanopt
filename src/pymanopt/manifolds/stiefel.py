from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.numerics import DummyNumericsBackendSingleton, NumericsBackend


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
        The formula for the exponential map can be found in [ZH2021]_.

        The Weingarten map is taken from [AMT2013]_.

        The default retraction used here is a first-order one based on
        the QR decomposition.
        To switch to a second-order polar retraction, use ``Stiefel(n, p, k=k,
        retraction="polar")``.
    """

    def __init__(
        self,
        n: int,
        p: int,
        *,
        k: int = 1,
        retraction: str = "qr",
        backend: NumericsBackend = DummyNumericsBackendSingleton,
    ):
        self._n = n
        self._p = p
        self._k = k

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )

        if k == 1:
            name = f"Stiefel manifold St({n}, {p})"
        elif k >= 2:
            name = f"Product Stiefel manifold St({n}, {p})^{k}"
        else:
            raise ValueError(f"Invalid value for k: {k} (should be >= 1)")

        dimension = int(k * (n * p - p * (p + 1) / 2))
        super().__init__(name, dimension, backend=backend)

        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> int:
        return self._p

    @property
    def k(self) -> int:
        return self._k

    @property
    def typical_dist(self):
        return self.backend.sqrt(self._p * self._k)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self.backend.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return vector - point @ self.backend.sym(
            self.backend.transpose(point) @ vector
        )

    to_tangent_space = projection

    def weingarten(self, point, tangent_vector, normal_vector):
        return -tangent_vector @ self.backend.transpose(
            point
        ) @ normal_vector - point @ self.backend.sym(
            self.backend.transpose(tangent_vector) @ normal_vector
        )

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        a = point + tangent_vector
        point, _ = self.backend.linalg_qr(a)
        return point

    def _retraction_polar(self, point, tangent_vector):
        Y = point + tangent_vector
        u, _, vt = self.backend.linalg_svd(Y, full_matrices=False)
        return u @ vt

    def norm(self, point, tangent_vector):
        return self.backend.linalg_norm(tangent_vector)

    def random_point(self):
        point, _ = self.backend.linalg_qr(
            self.backend.random_normal(size=(self._k, self._n, self._p))
        )
        if self._k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        vector = self.backend.random_normal(size=point.shape)
        vector = self.projection(point, vector)
        return vector / self.backend.linalg_norm(vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def exp(self, point, tangent_vector):
        bk = self.backend
        pt_tv = bk.transpose(point) @ tangent_vector
        identity = bk.squeeze(bk.multieye(self.k, self.p))

        a = bk.concatenate([point, tangent_vector], -1)
        b = bk.linalg_expm(
            bk.concatenate(
                [
                    bk.concatenate(
                        [
                            pt_tv,
                            -bk.transpose(tangent_vector) @ tangent_vector,
                        ],
                        -1,
                    ),
                    bk.concatenate([identity, pt_tv], -1),
                ],
                -2,
            )
        )[..., : self.p]
        c = bk.linalg_expm(-pt_tv)
        # breakpoint()
        return a @ (b @ c)

    def zero_vector(self, point):
        return self.backend.squeeze(
            self.backend.zeros((self.k, self.n, self.p))
        )
