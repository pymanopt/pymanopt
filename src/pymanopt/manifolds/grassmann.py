from pymanopt.backends import Backend, DummyBackendSingleton
from pymanopt.manifolds.manifold import Manifold


class _GrassmannBase(Manifold):
    @property
    def typical_dist(self):
        return self.backend.sqrt(self._p * self._k)

    def norm(self, point, tangent_vector):
        return self.backend.linalg_norm(tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def zero_vector(self, point):
        zero = self.backend.zeros((self._k, self._n, self._p))
        if self._k == 1:
            return zero[0]
        return zero

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)

    def to_tangent_space(self, point, vector):
        return self.projection(point, vector)


class Grassmann(_GrassmannBase):
    r"""The Grassmann manifold.

    This is the manifold of subspaces of dimension ``p`` of a real vector space
    of dimension ``n``.
    The optional argument ``k`` allows to optimize over the product of ``k``
    Grassmann manifolds.
    Elements are represented as ``n x p`` matrices if ``k == 1``, and as ``k x
    n x p`` arrays if ``k > 1``.

    Args:
        n: Dimension of the ambient space.
        p: Dimension of the subspaces.
        k: The number of elements in the product.

    Note:
        The geometry assumed here is the one obtained by treating the
        Grassmannian as a Riemannian quotient manifold of the Stiefel manifold
        (see also :class:`pymanopt.manifolds.stiefel.Stiefel`)
        with the orthogonal group :math:`\O(p) = \set{\vmQ \in \R^{p \times p}
        : \transp{\vmQ}\vmQ = \vmQ\transp{\vmQ} = \Id_p}`.
    """

    def __init__(
        self,
        n: int,
        p: int,
        *,
        k: int = 1,
        backend: Backend = DummyBackendSingleton,
    ):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )

        if k == 1:
            name = f"Grassmann manifold Gr({n}, {p})"
        elif k >= 2:
            name = f"Product Grassmann manifold Gr({n}, {p})^{k}"
        else:
            raise ValueError(f"Invalid value for k: {k} (should be >= 1)")

        dimension = int(k * (n * p - p**2))
        super().__init__(name, dimension, backend=backend)

    def dist(self, point_a, point_b):
        bk = self.backend
        s = bk.linalg_svdvals(bk.transpose(point_a) @ point_b)
        s = bk.where(s > 1.0, 1.0, s)
        s = bk.arccos(s)
        return bk.linalg_norm(s)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self.backend.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return vector - point @ (self.backend.transpose(point) @ vector)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hessian)
        XtG = self.backend.transpose(point) @ euclidean_gradient
        HXtG = tangent_vector @ XtG
        return PXehess - HXtG

    def retraction(self, point, tangent_vector):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X + G.
        u, _, vt = self.backend.linalg_svd(
            point + tangent_vector, full_matrices=False
        )
        return u @ vt

    def random_point(self):
        q, _ = self.backend.linalg_qr(
            self.backend.random_normal(size=(self._k, self._n, self._p))
        )
        if self._k == 1:
            return q[0]
        return q

    def random_tangent_vector(self, point):
        tangent_vector = self.backend.random_normal(size=point.shape)
        tangent_vector = self.projection(point, tangent_vector)
        return tangent_vector / self.backend.linalg_norm(tangent_vector)

    def exp(self, point, tangent_vector):
        u, s, vt = self.backend.linalg_svd(tangent_vector, full_matrices=False)
        cos_s = self.backend.expand_dims(self.backend.cos(s), -2)
        sin_s = self.backend.expand_dims(self.backend.sin(s), -2)

        Y = (
            point @ (self.backend.transpose(vt) * cos_s) @ vt
            + (u * sin_s) @ vt
        )

        # From numerical experiments, it seems necessary to re-orthonormalize.
        # This is quite expensive.
        q, _ = self.backend.linalg_qr(Y)
        return q

    def log(self, point_a, point_b):
        bk = self.backend
        ytx = bk.transpose(point_b) @ point_a
        At = bk.transpose(point_b) - ytx @ bk.transpose(point_a)
        Bt = bk.linalg_solve(ytx, At)
        u, s, vt = bk.linalg_svd(bk.transpose(Bt), full_matrices=False)
        arctan_s = bk.expand_dims(bk.arctan(s), -2)
        return (u * arctan_s) @ vt


class ComplexGrassmann(_GrassmannBase):
    r"""The complex Grassmann manifold.

    This is the manifold of subspaces of dimension ``p`` of complex
    vector space of dimension ``n``.
    The optional argument ``k`` allows to optimize over the product of ``k``
    complex Grassmannians.
    Elements are represented as ``n x p`` matrices if ``k == 1``, and as ``k x
    n x p`` arrays if ``k > 1``.

    Args:
        n: Dimension of the ambient space.
        p: Dimension of the subspaces.
        k: The number of elements in the product.

    Note:
        Similar to :class:`Grassmann`, the complex Grassmannian is treated
        as a Riemannian quotient manifold of the complex Stiefel manifold
        with the unitary group :math:`\U(p) = \set{\vmU \in \R^{p \times p}
        : \transp{\vmU}\vmU = \vmU\transp{\vmU} = \Id_p}`.
    """

    IS_COMPLEX = True

    def __init__(
        self,
        n: int,
        p: int,
        *,
        k: int = 1,
        backend: Backend = DummyBackendSingleton,
    ):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )

        if k == 1:
            name = f"Complex Grassmann manifold Gr({n}, {p})"
        elif k >= 2:
            name = f"Product complex Grassmann manifold Gr({n}, {p})^{k}"
        else:
            raise ValueError(f"Invalid value for k: {k} (should be >= 1)")

        dimension = int(2 * k * (n * p - p**2))
        super().__init__(name, dimension, backend=backend)

    def dist(self, point_a, point_b):
        bk = self.backend
        s = bk.linalg_svdvals(
            bk.conjugate_transpose(point_a) @ point_b,
        )
        s = bk.where(s > 1.0, 1.0, s)
        s = bk.arccos(s)
        return bk.linalg_norm(self.backend.real(s))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        bk = self.backend
        return bk.real(
            bk.tensordot(
                bk.conjugate(tangent_vector_a),
                tangent_vector_b,
                bk.ndim(tangent_vector_a),
            )
        )

    def projection(self, point, vector):
        return (
            vector - point @ self.backend.conjugate_transpose(point) @ vector
        )

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hessian)
        XHG = self.backend.conjugate_transpose(point) @ euclidean_gradient
        HXHG = tangent_vector @ XHG
        return PXehess - HXHG

    def retraction(self, point, tangent_vector):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        u, _, vh = self.backend.linalg_svd(
            point + tangent_vector, full_matrices=False
        )
        return u @ vh

    def random_point(self):
        q, _ = self.backend.linalg_qr(
            self.backend.random_normal(size=(self._k, self._n, self._p))
        )
        if self._k == 1:
            return q[0]
        return q

    def random_tangent_vector(self, point):
        tangent_vector = self.backend.random_normal(size=point.shape)
        tangent_vector = self.projection(point, tangent_vector)
        return tangent_vector / self.backend.linalg_norm(tangent_vector)

    def exp(self, point, tangent_vector):
        U, S, VH = self.backend.linalg_svd(tangent_vector, full_matrices=False)
        cos_S = self.backend.expand_dims(self.backend.cos(S), -2)
        sin_S = self.backend.expand_dims(self.backend.sin(S), -2)
        Y = (
            point @ (self.backend.conjugate_transpose(VH) * cos_S) @ VH
            + (U * sin_S) @ VH
        )

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        q, _ = self.backend.linalg_qr(Y)
        return q

    def log(self, point_a, point_b):
        YHX = self.backend.conjugate_transpose(point_b) @ point_a
        AH = self.backend.conjugate_transpose(
            point_b
        ) - YHX @ self.backend.conjugate_transpose(point_a)
        BH = self.backend.linalg_solve(YHX, AH)
        U, S, VH = self.backend.linalg_svd(
            self.backend.conjugate_transpose(BH), full_matrices=False
        )
        arctan_S = self.backend.expand_dims(self.backend.arctan(S), -2)
        return (U * arctan_S) @ VH
