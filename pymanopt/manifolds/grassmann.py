"""
Factory class for the Grassmann manifold. This is the manifold of p-
dimensional subspaces of n dimensional real vector space. Initiation requires
the dimensions n, p to be specified. Optional argument k allows the user
to optimize over the product of k Grassmanns.

Elements are represented as n x p matrices (if k == 1), and as k x n x p
matrices if k > 1 (Note that this is different to manopt!).
"""
import numpy as np
from pymanopt.tools.multi import multiprod, multitransp
from manifold import Manifold

class Grassmann(Manifold):

    def __init__(self, height, width, k = 1):
        # Check that n is greater than or equal to p
        if height < width or width < 1: raise ValueError("Need n >= p >= 1. "
            "Values supplied were n = %d and p = %d." % (height, width))
        if k < 1: raise ValueError("Need k >= 1. Value supplied was k = %d."
                % k)
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
            self._name = "Product Grassmann manifold Gr(%d, %d)^%d" % (self._n,
                self._p, self._k)

    @property
    def dim(self):
        return self._dim

    @property
    def name(self):
        return self._name

    # Geodesic distance for Grassmann
    def dist(self, X, Y):
        if self._k == 1:
            u, s, v = np.linalg.svd(np.dot(X.T,Y))
            s[s>1] = 1
            s = np.arccos(s)
            return np.linalg.norm(s)
        else:
            XtY = multiprod(multitransp(X), Y)
            square_d = 0
            for i in xrange(k):
                u, s, v = np.linalg.svd(XtY[i])
                square_d = square_d + np.linalg.norm(np.arccos(s))**2
            return np.sqrt(square_d)

    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the Grassmann this is the Frobenius inner product.
        np.tensordot(G,H, axes=G.ndim)

    def proj(self, X, U):
        # Project into the tangent space. Usually the same as egrad2rgrad
        if self._k == 1:
            UNew = U - np.dot(X, np.dot(X.T, U))
            return UNew
        else:
            UNew = U - multiprod(X, multiprod(multitransp(X), U))
            return UNew

    egrad2rgrad = proj

    def ehess2rhess(self, X, Hess):
        # Convert Euclidean hessian into Riemannian hessian.
        raise NotImplementedError()

    # Retract to the Grassmann using the qr decomposition of X + G. This
    # retraction may need to be changed - see manopt grassmannfactory.m. For now
    # it is identical to the Stiefel retraction.
    def retr(self, X, G):
        if self._k == 1:
            # Calculate 'thin' qr decomposition of X + G
            q, r = np.linalg.qr(X + G)
            # Unflip any flipped signs
            XNew = np.dot(q, np.diag(np.sign(np.sign(np.diag(r))+.5)))
            return XNew
        else:
            XNew = X + G
            for i in xrange(self._k):
                q, r = np.linalg.qr(Y[i])
                XNew[i] = np.dot(q, np.diag(np.sign(np.sign(np.diag(r))+.5)))

    def norm(self, X, G):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.linalg.norm(G)

    # Generate random Grassmann point using qr of random normally distributed
    # matrix.
    def rand(self):
        if self._k == 1:
            X = np.random.randn(self._n,self._p)
            q, r = np.linalg.qr(X)
            return q
        else:
            X = np.zeros((self._k, self._n, self._p))
            for i in xrange(self._k):
                X[i], r = np.linalg.qr(np.random.randn(self._n, self._p))
            return X
