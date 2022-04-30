import numpy as np
from numpy.linalg import svd

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multihconj, multiprod, multitransp


class _GrassmannBase(Manifold):
    @property
    def typical_dist(self):
        return np.sqrt(self._p * self._k)

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def zero_vector(self, point):
        if self._k == 1:
            return np.zeros((self._n, self._p))
        return np.zeros((self._k, self._n, self._p))

    def egrad2rgrad(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)


class Grassmann(_GrassmannBase):
    """The Grassmannian.

    This is the manifold of p-dimensional subspaces of n dimensional real
    vector space.
    The optional argument k allows the user to optimize over the product of k
    Grassmannians.
    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1.
    """

    def __init__(self, n, p, k=1):
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
        s = svd(multiprod(multitransp(point_a), point_b), compute_uv=False)
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(s)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def projection(self, point, vector):
        return vector - multiprod(point, multiprod(multitransp(point), vector))

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hvp)
        XtG = multiprod(multitransp(point), euclidean_gradient)
        HXtG = multiprod(tangent_vector, XtG)
        return PXehess - HXtG

    def retraction(self, point, tangent_vector):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        u, _, vt = svd(point + tangent_vector, full_matrices=False)
        return multiprod(u, vt)

    # Generate random Grassmann point using qr of random normally distributed
    # matrix.
    def random_point(self):
        if self._k == 1:
            X = np.random.normal(size=(self._n, self._p))
            q, _ = np.linalg.qr(X)
            return q

        X = np.zeros((self._k, self._n, self._p))
        for i in range(self._k):
            X[i], _ = np.linalg.qr(np.random.normal(size=(self._n, self._p)))
        return X

    def random_tangent_vector(self, point):
        tangent_vector = np.random.normal(size=point.shape)
        tangent_vector = self.projection(point, tangent_vector)
        return tangent_vector / np.linalg.norm(tangent_vector)

    def exp(self, point, tangent_vector):
        u, s, vt = svd(tangent_vector, full_matrices=False)
        cos_s = np.expand_dims(np.cos(s), -2)
        sin_s = np.expand_dims(np.sin(s), -2)

        Y = multiprod(
            multiprod(point, multitransp(vt) * cos_s), vt
        ) + multiprod(u * sin_s, vt)

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        if self._k == 1:
            Y, unused = np.linalg.qr(Y)
            return Y

        for i in range(self._k):
            Y[i], unused = np.linalg.qr(Y[i])
        return Y

    def log(self, point_a, point_b):
        ytx = multiprod(multitransp(point_b), point_a)
        At = multitransp(point_b) - multiprod(ytx, multitransp(point_a))
        Bt = np.linalg.solve(ytx, At)
        u, s, vt = svd(multitransp(Bt), full_matrices=False)
        arctan_s = np.expand_dims(np.arctan(s), -2)
        return multiprod(u * arctan_s, vt)


class ComplexGrassmann(_GrassmannBase):
    """The complex Grassmannian.

    This is the manifold of p-dimensional subspaces of n dimensional complex
    vector space.
    The optional argument k allows the user to optimize over the product of k
    Grassmannians.
    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1.
    """

    def __init__(self, n, p, k=1):
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
        s = np.linalg.svd(
            multiprod(multihconj(point_a), point_b), compute_uv=False
        )
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
        return vector - multiprod(point, multiprod(multihconj(point), vector))

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        PXehess = self.projection(point, euclidean_hvp)
        XHG = multiprod(multihconj(point), euclidean_gradient)
        HXHG = multiprod(tangent_vector, XHG)
        return PXehess - HXHG

    def retraction(self, point, tangent_vector):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        u, _, vh = np.linalg.svd(point + tangent_vector, full_matrices=False)
        return multiprod(u, vh)

    def random_point(self):
        if self._k == 1:
            point, _ = np.linalg.qr(
                (
                    np.random.normal(size=(self._n, self._p))
                    + 1j * np.random.normal(size=(self._n, self._p))
                )
            )
            return point

        point = np.zeros((self._k, self._n, self._p), np.complex_)
        for i in range(self._k):
            point[i], _ = np.linalg.qr(
                (
                    np.random.normal(size=(self._n, self._p))
                    + 1j * np.random.normal(size=(self._n, self._p))
                )
            )
        return point

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
        Y = multiprod(
            multiprod(point, multihconj(VH) * cos_S), VH
        ) + multiprod(U * sin_S, VH)

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        if self._k == 1:
            Y, _ = np.linalg.qr(Y)
            return Y

        for i in range(self._k):
            Y[i], _ = np.linalg.qr(Y[i])
        return Y

    def log(self, point_a, point_b):
        YHX = multiprod(multihconj(point_b), point_a)
        AH = multihconj(point_b) - multiprod(YHX, multihconj(point_a))
        BH = np.linalg.solve(YHX, AH)
        U, S, VH = np.linalg.svd(multihconj(BH), full_matrices=False)
        arctan_S = np.expand_dims(np.arctan(S), -2)
        return multiprod(U * arctan_S, VH)
