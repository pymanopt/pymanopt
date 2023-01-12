import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multihconj, multiqr, multitransp


class _GrassmannBase(Manifold):
    @property
    def typical_dist(self):
        return np.sqrt(self._p * self._k)

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def zero_vector(self, point):
        zero = np.zeros((self._k, self._n, self._p))
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

    def __init__(self, n: int, p: int, *, k: int = 1):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )
        if k < 1:
            raise ValueError(f"Need k >= 1. Value supplied was k = {k}")

        if k == 1:
            name = f"Grassmann manifold Gr({n},{p})"
        elif k >= 2:
            name = f"Product Grassmann manifold Gr({n},{p})^{k}"
        dimension = int(k * (n * p - p**2))
        super().__init__(name, dimension)

    def dist(self, point_a, point_b):
        s = np.linalg.svd(multitransp(point_a) @ point_b, compute_uv=False)
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(s)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return vector - point @ (multitransp(point) @ vector)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hessian)
        XtG = multitransp(point) @ euclidean_gradient
        HXtG = tangent_vector @ XtG
        return PXehess - HXtG

    def retraction(self, point, tangent_vector):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X + G.
        u, _, vt = np.linalg.svd(point + tangent_vector, full_matrices=False)
        return u @ vt

    def random_point(self):
        q, _ = multiqr(np.random.normal(size=(self._k, self._n, self._p)))
        if self._k == 1:
            return q[0]
        return q

    def random_tangent_vector(self, point):
        tangent_vector = np.random.normal(size=point.shape)
        tangent_vector = self.projection(point, tangent_vector)
        return tangent_vector / np.linalg.norm(tangent_vector)

    def exp(self, point, tangent_vector):
        u, s, vt = np.linalg.svd(tangent_vector, full_matrices=False)
        cos_s = np.expand_dims(np.cos(s), -2)
        sin_s = np.expand_dims(np.sin(s), -2)

        Y = point @ (multitransp(vt) * cos_s) @ vt + (u * sin_s) @ vt

        # From numerical experiments, it seems necessary to re-orthonormalize.
        # This is quite expensive.
        q, _ = multiqr(Y)
        return q

    def log(self, point_a, point_b):
        ytx = multitransp(point_b) @ point_a
        At = multitransp(point_b) - ytx @ multitransp(point_a)
        Bt = np.linalg.solve(ytx, At)
        u, s, vt = np.linalg.svd(multitransp(Bt), full_matrices=False)
        arctan_s = np.expand_dims(np.arctan(s), -2)
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

    def __init__(self, n: int, p: int, *, k: int = 1):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )
        if k < 1:
            raise ValueError(f"Need k >= 1. Value supplied was k = {k}")

        if k == 1:
            name = f"Complex Grassmann manifold Gr({n},{p})"
        elif k >= 2:
            name = f"Product complex Grassmann manifold Gr({n},{p})^{k}"
        dimension = int(2 * k * (n * p - p**2))
        super().__init__(name, dimension)

    def dist(self, point_a, point_b):
        s = np.linalg.svd(multihconj(point_a) @ point_b, compute_uv=False)
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(np.real(s))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.real(
            np.tensordot(
                np.conjugate(tangent_vector_a),
                tangent_vector_b,
                axes=tangent_vector_a.ndim,
            )
        )

    def projection(self, point, vector):
        return vector - point @ multihconj(point) @ vector

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hessian)
        XHG = multihconj(point) @ euclidean_gradient
        HXHG = tangent_vector @ XHG
        return PXehess - HXHG

    def retraction(self, point, tangent_vector):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        u, _, vh = np.linalg.svd(point + tangent_vector, full_matrices=False)
        return u @ vh

    def random_point(self):
        q, _ = multiqr(
            np.random.normal(size=(self._k, self._n, self._p))
            + 1j * np.random.normal(size=(self._k, self._n, self._p))
        )
        if self._k == 1:
            return q[0]
        return q

    def random_tangent_vector(self, point):
        tangent_vector = np.random.normal(
            size=point.shape
        ) + 1j * np.random.normal(size=point.shape)
        tangent_vector = self.projection(point, tangent_vector)
        return tangent_vector / np.linalg.norm(tangent_vector)

    def exp(self, point, tangent_vector):
        U, S, VH = np.linalg.svd(tangent_vector, full_matrices=False)
        cos_S = np.expand_dims(np.cos(S), -2)
        sin_S = np.expand_dims(np.sin(S), -2)
        Y = point @ (multihconj(VH) * cos_S) @ VH + (U * sin_S) @ VH

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        q, _ = multiqr(Y)
        return q

    def log(self, point_a, point_b):
        YHX = multihconj(point_b) @ point_a
        AH = multihconj(point_b) - YHX @ multihconj(point_a)
        BH = np.linalg.solve(YHX, AH)
        U, S, VH = np.linalg.svd(multihconj(BH), full_matrices=False)
        arctan_S = np.expand_dims(np.arctan(S), -2)
        return (U * arctan_S) @ VH
