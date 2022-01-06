import numpy as np
from numpy.linalg import svd

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multiprod, multitransp


class Grassmann(Manifold):
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
        dimension = int(k * (n * p - p ** 2))
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self._p * self._k)

    # Geodesic distance for Grassmann
    def dist(self, X, Y):
        u, s, v = svd(multiprod(multitransp(X), Y))
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(s)

    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the Grassmann this is the Frobenius inner product.
        return np.tensordot(G, H, axes=G.ndim)

    def proj(self, X, U):
        return U - multiprod(X, multiprod(multitransp(X), U))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Convert Euclidean into Riemannian Hessian.
        PXehess = self.proj(X, ehess)
        XtG = multiprod(multitransp(X), egrad)
        HXtG = multiprod(H, XtG)
        return PXehess - HXtG

    def retr(self, X, G):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        u, s, vt = svd(X + G, full_matrices=False)
        return multiprod(u, vt)

    def norm(self, X, G):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.linalg.norm(G)

    # Generate random Grassmann point using qr of random normally distributed
    # matrix.
    def rand(self):
        if self._k == 1:
            X = np.random.randn(self._n, self._p)
            q, r = np.linalg.qr(X)
            return q

        X = np.zeros((self._k, self._n, self._p))
        for i in range(self._k):
            X[i], r = np.linalg.qr(np.random.randn(self._n, self._p))
        return X

    def randvec(self, X):
        U = np.random.randn(*np.shape(X))
        U = self.proj(X, U)
        U = U / np.linalg.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def exp(self, X, U):
        u, s, vt = svd(U, full_matrices=False)
        cos_s = np.expand_dims(np.cos(s), -2)
        sin_s = np.expand_dims(np.sin(s), -2)

        Y = multiprod(multiprod(X, multitransp(vt) * cos_s), vt) + multiprod(
            u * sin_s, vt
        )

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        if self._k == 1:
            Y, unused = np.linalg.qr(Y)
            return Y
        else:
            for i in range(self._k):
                Y[i], unused = np.linalg.qr(Y[i])
            return Y

    def log(self, X, Y):
        ytx = multiprod(multitransp(Y), X)
        At = multitransp(Y) - multiprod(ytx, multitransp(X))
        Bt = np.linalg.solve(ytx, At)
        u, s, vt = svd(multitransp(Bt), full_matrices=False)
        arctan_s = np.expand_dims(np.arctan(s), -2)

        U = multiprod(u * arctan_s, vt)
        return U

    def zerovec(self, X):
        if self._k == 1:
            return np.zeros((self._n, self._p))
        return np.zeros((self._k, self._n, self._p))
