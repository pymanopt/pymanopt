"""
Factory class for the Grassmann manifold. This is the manifold of p-
dimensional subspaces of n dimensional real vector space. Initiation requires
the dimensions n, p to be specified. Optional argument k allows the user
to optimize over the product of k Grassmanns.

Elements are represented as n x p matrices (if k == 1), and as k x n x p
matrices if k > 1 (Note that this is different to manopt!).
"""

#   I have chaned the retraction to one using the polar decomp as am now
#   implementing vector transport. See comment below (JT)

#   April 17, 2013 (NB) :
#       Retraction changed to the polar decomposition, so that the vector
#       transport is now correct, in the sense that it is compatible with
#       the retraction, i.e., transporting a tangent vector G from U to V
#       where V = Retr(U, H) will give Z, and transporting GQ from UQ to VQ
#       will give ZQ: there is no dependence on the representation, which
#       is as it should be. Notice that the polar factorization requires an
#       SVD whereas the qfactor retraction requires a QR decomposition,
#       which is cheaper. Hence, if the retraction happens to be a
#       bottleneck in your application and you are not using vector
#       transports, you may want to replace the retraction with a qfactor.

import numpy as np
from scipy.linalg import svd
from pymanopt.tools.multi import multiprod, multitransp, multisym
from pymanopt.manifolds.manifold import Manifold


class Grassmann(Manifold):
    def __init__(self, height, width, k=1):
        # Check that n is greater than or equal to p
        if height < width or width < 1:
            raise ValueError("Need n >= p >= 1. Values supplied were n = %d "
                             "and p = %d." % (height, width))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)
        # Set the dimensions of the Grassmann
        self._n = height
        self._p = width
        self._k = k

        # Set dimension
        self._dim = self._k*(self._n*self._p - self._p**2)

        # Set the name
        if k == 1:
            self._name = "Grassmann manifold Gr(%d, %d)" % (self._n, self._p)
        elif k >= 2:
            self._name = "Product Grassmann manifold Gr(%d, %d)^%d" % (
                self._n, self._p, self._k)

    @property
    def dim(self):
        return self._dim

    @property
    def name(self):
        return self._name

    @property
    def typicaldist(self):
        return np.sqrt(self._p * self._k)

    # Geodesic distance for Grassmann
    def dist(self, X, Y):
        if self._k == 1:
            u, s, v = np.linalg.svd(np.dot(X.T, Y))
            s[s > 1] = 1
            s = np.arccos(s)
            return np.linalg.norm(s)
        else:
            XtY = multiprod(multitransp(X), Y)
            square_d = 0
            for i in xrange(self._k):
                s = np.linalg.svd(XtY[i], compute_uv=False)
                # Ensure that -1 <= s <= 1
                s = np.fmin(s, [1])
                s = np.fmax(s, [-1])
                square_d = square_d + np.linalg.norm(np.arccos(s))**2
            return np.sqrt(square_d)

    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the Grassmann this is the Frobenius inner product.
        return np.tensordot(G, H, axes=G.ndim)

    def proj(self, X, U):
        return U - multiprod(X, multiprod(multitransp(X), U))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Convert Euclidean hessian into Riemannian hessian.
        PXehess = self.proj(X, ehess)
        XtG = multiprod(multitransp(X), egrad)
        HXtG = multiprod(H, XtG)
        return PXehess - HXtG

    def retr(self, X, G):
        if self._k == 1:
            # Calculate 'thin' qr decomposition of X + G
            # XNew, r = np.linalg.qr(X + G)

            # We do not need to worry about flipping signs of columns here,
            # since only the column space is important, not the actual
            # columns. Compare this with the Stiefel manifold.

            # Compute the polar factorization of Y = X+G
            u, s, v = svd(X + G, full_matrices=False)
            XNew = u.dot(v.T)
        else:
            XNew = np.zeros(np.shape(X))
            for i in xrange(self._k):
                # q, r = np.linalg.qr(Y[i])
                # XNew[i] = np.dot(q, np.diag(np.sign(np.sign(np.diag(r))+.5)))
                u, s, vt = svd(X[i] + G[i], full_matrices=False)
                XNew[i] = u.dot(vt)
        return XNew

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
        for i in xrange(self._k):
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
        if self._k == 1:
            u, s, vt = svd(U, full_matrices=False)
            cos_s = np.diag(np.cos(s))
            sin_s = np.diag(np.sin(s))
            Y = X.dot(vt.T).dot(cos_s).dot(vt) + u.dot(sin_s).dot(vt)
            # From numerical experiments, it seems necessary to
            # re-orthonormalize. This is overall quite expensive.
            Y, unused = np.linalg.qr(Y)
        else:
            Y = np.zeros((self._k, self._n, self._p))
            for i in xrange(self._k):
                u, s, vt = svd(U[i], full_matrices=False)
                cos_s = np.diag(np.cos(s))
                sin_s = np.diag(np.sin(s))
                Y[i] = X[i].dot(vt.T).dot(cos_s).dot(vt) + u.dot(sin_s).dot(vt)
                # From numerical experiments, it seems necessary to
                # re-orthonormalize. This is overall quite expensive.
                Y[i], unused = np.linalg.qr(Y[i])
        return Y

    def log(self, X, Y):
        if self._k == 1:
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)
        U = np.zeros((self._k, self._n, self._p))
        for i in xrange(self._k):
            x = X[i]
            y = Y[i]
            ytx = y.T.dot(x)
            At = y.T - ytx.dot(x.T)
            Bt = np.linalg.solve(ytx, At)
            u, s, vt = svd(Bt.T, full_matrices=False)
            U[i] = u.dot(np.diag(np.arctan(s))).dot(vt)
        if self._k == 1:
            U = U[0]
        return U
