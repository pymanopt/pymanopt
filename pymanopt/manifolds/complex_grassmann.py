import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multihconj, multiprod


class ComplexGrassmann(Manifold):
    """
    Factory class for the Grassmann manifold. This is the manifold of p-
    dimensional subspaces of n dimensional complex vector space. Initiation
    requires the dimensions n, p to be specified. Optional argument k
    allows the user to optimize over the product of k Grassmanns.

    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1 (Note that this is different to manopt!).
    """

    def __init__(self, n, p, k=1):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError("Need n >= p >= 1. Values supplied were n = %d "
                             "and p = %d." % (n, p))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)

        if k == 1:
            name = "Complex Grassmann manifold Gr({:d}, {:d})".format(n, p)
        elif k >= 2:
            name = ("Product complex Grassmann manifold Gr({:d}, {:d})^{:d}"
                    .format(n, p, k))
        dimension = int(2 * k * (n * p - p ** 2))
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self._p * self._k)

    def inner(self, X, G, H):
        return np.real(np.tensordot(np.conjugate(G), H, axes=G.ndim))

    def proj(self, X, U):
        return U - multiprod(X, multiprod(multihconj(X), U))

    def norm(self, X, G):
        return np.linalg.norm(G)

    def rand(self):
        if self._k == 1:
            q, _ = np.linalg.qr((np.random.randn(self._n, self._p)
                                 + 1j*np.random.randn(self._n, self._p)))
            return q

        X = np.zeros((self._k, self._n, self._p), np.complex_)
        for i in range(self._k):
            X[i], _ = np.linalg.qr((np.random.randn(self._n, self._p)
                                    + 1j*np.random.randn(self._n, self._p)))
        return X

    def randvec(self, X):
        U = np.random.randn(*np.shape(X)) + 1j*np.random.randn(*np.shape(X))
        U = self.proj(X, U)
        U = U / np.linalg.norm(U)
        return U

    def zerovec(self, X):
        if self._k == 1:
            return np.zeros((self._n, self._p))
        return np.zeros((self._k, self._n, self._p))

    def dist(self, X, Y):
        _, s, _ = np.linalg.svd(multiprod(multihconj(X), Y))
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(np.real(s))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        PXehess = self.proj(X, ehess)
        XHG = multiprod(multihconj(X), egrad)
        HXHG = multiprod(H, XHG)
        return PXehess - HXHG

    def retr(self, X, G):
        # Calculate 'thin' qr decomposition of X + G
        # XNew, r = np.linalg.qr(X + G)

        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        u, s, vh = np.linalg.svd(X + G, full_matrices=False)
        return multiprod(u, vh)

    def exp(self, X, U):
        U, S, VH = np.linalg.svd(U, full_matrices=False)
        cos_S = np.expand_dims(np.cos(S), -2)
        sin_S = np.expand_dims(np.sin(S), -2)
        Y = (multiprod(multiprod(X, multihconj(VH) * cos_S), VH)
             + multiprod(U * sin_S, VH))

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        if self._k == 1:
            Y, _ = np.linalg.qr(Y)
            return Y
        else:
            for i in range(self._k):
                Y[i], _ = np.linalg.qr(Y[i])
            return Y

    def log(self, X, Y):
        YHX = multiprod(multihconj(Y), X)
        AH = multihconj(Y) - multiprod(YHX, multihconj(X))
        BH = np.linalg.solve(YHX, AH)
        U, S, VH = np.linalg.svd(multihconj(BH), full_matrices=False)
        arctan_S = np.expand_dims(np.arctan(S), -2)
        U = multiprod(U * arctan_S, VH)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)
