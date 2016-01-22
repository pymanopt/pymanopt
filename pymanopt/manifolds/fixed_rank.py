import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from scipy.linalg import solve_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold


class SymFixedRankYY(Manifold):
    """
    Manifold of n-by-n symmetric positive semidefinite matrices of rank k.

    A point X on the manifold is parameterized as YY^T where Y is a matrix of
    size nxk. As such, X is symmetric, positive semidefinite. We restrict to
    full-rank Y's, such that X has rank exactly k. The point X is numerically
    represented by Y (this is more efficient than working with X, which may
    be big). Tangent vectors are represented as matrices of the same size as
    Y, call them Ydot, so that Xdot = Y Ydot' + Ydot Y. The metric is the
    canonical Euclidean metric on Y.

    Since for any orthogonal Q of size k, it holds that (YQ)(YQ)' = YY',
    we "group" all matrices of the form YQ in an equivalence class. The set
    of equivalence classes is a Riemannian quotient manifold, implemented
    here.

    Notice that this manifold is not complete: if optimization leads Y to be
    rank-deficient, the geometry will break down. Hence, this geometry should
    only be used if it is expected that the points of interest will have rank
    exactly k. Reduce k if that is not the case.

    An alternative, complete, geometry for positive semidefinite matrices of
    rank k is described in Bonnabel and Sepulchre 2009, "Riemannian Metric
    and Geometric Mean for Positive Semidefinite Matrices of Fixed Rank",
    SIAM Journal on Matrix Analysis and Applications.


    The geometry implemented here is the simplest case of the 2010 paper:
    M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
    "Low-Rank Optimization on the Cone of Positive Semidefinite Matrices".
    Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
    """
    def __init__(self, n, k):
        self._n = n
        self._k = k

        self._name = ("YY' quotient manifold of {:d}x{:d} psd matrices of "
                      "rank {:d}".format(n, n, k))

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        n = self._n
        k = self._k
        return k * n - k * (k - 1) / 2

    @property
    def typicaldist(self):
        return 10 + self._k

    def inner(self, Y, U, V):
        return float(np.tensordot(np.asmatrix(U), np.asmatrix(V)))

    def norm(self, Y, U):
        return la.norm(U, "fro")

    def dist(self, U, V):
        raise NotImplementedError

    def proj(self, Y, H):
        # Projection onto the horizontal space
        YtY = Y.T.dot(Y)
        SS = YtY
        AS = Y.T.dot(H) - H.T.dot(Y)
        Omega = lyap(SS, -AS)
        return H - Y.dot(Omega)

    tangent = proj

    def egrad2rgrad(self, Y, H):
        return H

    def ehess2rhess(self, Y, egrad, ehess, U):
        return self.proj(Y, ehess)

    def exp(self, Y, U):
        import warnings
        warnings.warn("Exponential map for symmetric, fixed-rank "
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(Y, U)

    def retr(self, Y, U):
        return Y + U

    def log(self, Y, U):
        raise NotImplementedError

    def rand(self):
        return rnd.randn(self._n, self._k)

    def randvec(self, Y):
        H = self.rand()
        P = self.proj(Y, H)
        return self._normalize(P)

    def transp(self, Y, Z, U):
        return self.proj(Z, U)

    def _normalize(self, Y):
        return Y / self.norm(None, Y)
