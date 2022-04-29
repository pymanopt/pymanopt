import numpy as np
from numpy import linalg as la
from scipy.linalg import solve_continuous_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold, RetrAsExpMixin


class _PSDFixedRank(Manifold, RetrAsExpMixin):
    def __init__(self, n, k, name, dimension):
        self._n = n
        self._k = k
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return 10 + self._k

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return la.norm(tangent_vector, "fro")

    def projection(self, point, vector):
        YtY = point.T @ point
        AS = point.T @ vector - vector.T @ point
        Omega = lyap(YtY, AS)
        return vector - point @ Omega

    def egrad2rgrad(self, point, euclidean_gradient):
        return euclidean_gradient

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        return self.projection(point, euclidean_hvp)

    def retraction(self, point, tangent_vector):
        return point + tangent_vector

    def rand(self):
        return np.random.randn(self._n, self._k)

    def random_tangent_vector(self, point):
        random_vector = self.rand()
        tangent_vector = self.projection(point, random_vector)
        return self._normalize(tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def _normalize(self, array):
        return array / self.norm(None, array)

    def zero_vector(self, point):
        return np.zeros((self._n, self._k))


class PSDFixedRank(_PSDFixedRank):
    """Manifold of fixed-rank positive semidefinite (PSD) matrices.

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
        name = f"Quotient manifold of {n}x{n} psd matrices of rank {k}"
        dimension = int(k * n - k * (k - 1) / 2)
        super().__init__(n, k, name, dimension)


class PSDFixedRankComplex(_PSDFixedRank):
    """Manifold of fixed-rank Hermitian positive semidefinite (PSD) matrices.

    Manifold of n-by-n complex Hermitian positive semidefinite matrices of
    fixed rank k. This follows the quotient geometry described
    in Sarod Yatawatta's 2013 paper:
    "Radio interferometric calibration using a Riemannian manifold", ICASSP.

    Paper link: http://dx.doi.org/10.1109/ICASSP.2013.6638382.

    A point X on the manifold M is parameterized as YY^*, where Y is a
    complex matrix of size nxk of full rank. For any point Y on the manifold M,
    given any kxk complex unitary matrix U, we say Y*U  is equivalent to Y,
    i.e., YY^* does not change. Therefore, M is the set of equivalence
    classes and is a Riemannian quotient manifold C^{nk}/U(k)
    where C^{nk} is the set of all complex matrix of size nxk of full rank.
    The metric is the usual real-trace inner product, that is,
    it is the usual metric for the complex plane identified with R^2.

    Notice that this manifold is not complete: if optimization leads Y to be
    rank-deficient, the geometry will break down. Hence, this geometry should
    only be used if it is expected that the points of interest will have rank
    exactly k. Reduce k if that is not the case.
    """

    def __init__(self, n, k):
        name = f"Quotient manifold of Hermitian {n}x{n} matrices of rank {k}"
        dimension = 2 * k * n - k * k
        super().__init__(n, k, name, dimension)

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return (
            2
            * np.tensordot(
                tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
            ).real
        )

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner(point, tangent_vector, tangent_vector))

    def dist(self, point_a, point_b):
        s, _, d = la.svd(point_b.T.conj() @ point_a)
        e = point_a - point_b @ s @ d
        return self.inner(None, e, e) / 2

    def rand(self):
        rand_ = super().rand
        return rand_() + 1j * rand_()


class Elliptope(Manifold, RetrAsExpMixin):
    """Manifold of fixed-rank PSD matrices with unit diagonal elements.

    A point X on the manifold is parameterized as YY^T where Y is a matrix of
    size nxk. As such, X is symmetric, positive semidefinite. We restrict to
    full-rank Y's, such that X has rank exactly k. The point X is numerically
    represented by Y (this is more efficient than working with X, which may be
    big). Tangent vectors are represented as matrices of the same size as Y,
    call them Ydot, so that Xdot = Y Ydot' + Ydot Y and diag(Xdot) == 0. The
    metric is the canonical Euclidean metric on Y.

    The diagonal constraints on X (X(i, i) == 1 for all i) translate to
    unit-norm constraints on the rows of Y: norm(Y(i, :)) == 1 for all i.  The
    set of such Y's forms the oblique manifold. But because for any orthogonal
    Q of size k, it holds that (YQ)(YQ)' = YY', we "group" all matrices of the
    form YQ in an equivalence class. The set of equivalence classes is a
    Riemannian quotient manifold, implemented here.

    Note that this geometry formally breaks down at rank-deficient Y's.  This
    does not appear to be a major issue in practice when optimization
    algorithms converge to rank-deficient Y's, but convergence theorems no
    longer hold. As an alternative, you may use the oblique manifold (it has
    larger dimension, but does not break down at rank drop.)

    The geometry is taken from the 2010 paper:
    M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
    "Low-Rank Optimization on the Cone of Positive Semidefinite Matrices".
    Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
    """

    def __init__(self, n, k):
        self._n = n
        self._k = k

        name = (
            f"Quotient manifold of {n}x{n} psd matrices of rank {k} "
            "with unit diagonal elements"
        )
        dimension = int(n * (k - 1) - k * (k - 1) / 2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return 10 * self._k

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner(point, tangent_vector, tangent_vector))

    def projection(self, point, vector):
        eta = self._project_rows(point, vector)
        YtY = point.T @ point
        AS = point.T @ eta - vector.T @ point
        Omega = lyap(YtY, -AS)
        return eta - point @ (Omega - Omega.T) / 2

    def retraction(self, point, tangent_vector):
        return self._normalize_rows(point + tangent_vector)

    def egrad2rgrad(self, point, euclidean_gradient):
        return self._project_rows(point, euclidean_gradient)

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        scaling_grad = (euclidean_gradient * point).sum(axis=1)
        hess = euclidean_hvp - tangent_vector * scaling_grad[:, np.newaxis]
        scaling_hess = (
            tangent_vector * euclidean_gradient + point * euclidean_hvp
        ).sum(axis=1)
        hess -= point * scaling_hess[:, np.newaxis]
        return self.projection(point, hess)

    def rand(self):
        return self._normalize_rows(np.random.randn(self._n, self._k))

    def random_tangent_vector(self, point):
        tangent_vector = self.projection(point, self.rand())
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def _normalize_rows(self, array):
        """Return an l2-row-normalized copy of an array."""
        return array / la.norm(array, axis=1)[:, np.newaxis]

    def _project_rows(self, point, vector):
        """Orthogonal projection of each row of H to the tangent space at the
        corresponding row of X, seen as a point on a sphere.
        """
        # Compute the inner product between each vector H[i, :] with its root
        # point Y[i, :], i.e., Y[i, :].T * H[i, :]. Returns a row vector.
        inner_products = (point * vector).sum(axis=1)
        return vector - point * inner_products[:, np.newaxis]

    def zero_vector(self, point):
        return np.zeros((self._n, self._k))
