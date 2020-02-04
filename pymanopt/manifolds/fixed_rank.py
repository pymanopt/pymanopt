"""
Module containing manifolds of fixed rank matrices.
"""
import numpy as np

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.tools import ndarraySequenceMixin


class FixedRankEmbedded(EuclideanEmbeddedSubmanifold):
    """
    Note: Currently not compatible with the second order TrustRegions solver.
    Should be fixed soon.

    Manifold of m-by-n real matrices of fixed rank k. This follows the
    embedded geometry described in Bart Vandereycken's 2013 paper:
    "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    For efficiency purposes, Pymanopt does not represent points on this
    manifold explicitly using m x n matrices, but instead implicitly using
    a truncated singular value decomposition. Specifically, a point is
    represented by a tuple (u, s, vt) of three numpy arrays. The arrays u,
    s and vt have shapes (m, k), (k,) and (k, n) respectively, and the low
    rank matrix which they represent can be recovered by the matrix product
    u * diag(s) * vt.

    For example, to optimize over the space of 5 by 4 matrices with rank 3,
    we would need to
    >>> import pymanopt.manifolds
    >>> manifold = pymanopt.manifolds.FixedRankEmbedded(5, 4, 3)

    Then the shapes will be as follows:
    >>> x = manifold.rand()
    >>> x[0].shape
    (5, 3)
    >>> x[1].shape
    (3, 3)
    >>> x[2].shape
    (4, 3)

    and the full matrix can be recovered using the matrix product
    x[0] * x[1] * x[2].T:
    >>> import numpy as np
    >>> X = x[0] @ x[1] @ x[2].T

    Tangent vectors are represented as a tuple (Up, M, Vp). The matrices Up
    (m x k) and Vp (n x k) obey Up.T @ U = 0 and Vp.T @ V = 0.
    The matrix M (k x k) is arbitrary. Such a structure corresponds to the
    following tangent vector in the ambient space of m x n matrices:
      Z = U @ M @ V.T + Up @ V.T + U @ Vp.T
    where (U, S, V) is the current point and (Up, M, Vp) is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as m x n matrices. If
    these are low-rank, they may also be represented as structures with
    U, S, V fields, such that Z = U @ S @ V.T. There are no restrictions on
    what U, S and V are, as long as their product as indicated yields a real,
    m x n matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space R^(m x n) equipped with the usual trace (Frobenius) inner product.


    Please cite the Pymanopt paper as well as the research paper:
        @Article{vandereycken2013lowrank,
          Title   = {Low-rank matrix completion by {Riemannian} optimization},
          Author  = {Vandereycken, B.},
          Journal = {SIAM Journal on Optimization},
          Year    = {2013},
          Number  = {2},
          Pages   = {1214--1236},
          Volume  = {23},
          Doi     = {10.1137/110845768}
        }

    This file is based on fixedrankembeddedfactory from Manopt: www.manopt.org.
    Ported by: Jamie Townsend, Sebastian Weichwald
    Original author: Nicolas Boumal, Dec. 30, 2012.
    """

    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k
        self._stiefel_m = Stiefel(m, k)
        self._stiefel_n = Stiefel(n, k)

        name = ("Manifold of {m}-by-{n} matrices with rank {k} and embedded "
                "geometry".format(m=m, n=n, k=k))
        dimension = (m + n - k) * k
        super().__init__(name, dimension, point_layout=3)

    @property
    def typicaldist(self):
        return self.dim

    def inner(self, X, G, H):
        return np.sum(np.tensordot(a, b) for (a, b) in zip(G, H))

    def _apply_ambient(self, Z, W):
        """
        For a given ambient vector Z, given as a tuple (U, S, V) such that
        Z = U @ S @ V.T, applies it to a matrix W to calculate the matrix
        product ZW.
        """
        if isinstance(Z, (list, tuple)):
            return Z[0] @ Z[1] @ Z[2].T @ W
        return Z @ W

    def _apply_ambient_transpose(self, Z, W):
        """
        Same as apply_ambient, but applies Z.T to W.
        """
        if isinstance(Z, (list, tuple)):
            return Z[2] @ Z[1].T @ Z[0].T @ W
        return Z.T @ W

    def proj(self, X, Z):
        """
        Note that Z must either be an m x n matrix from the ambient space, or
        else a tuple (Uz, Sz, Vz), where Uz * Sz * Vz is in the ambient space
        (of low-rank matrices).

        This function then returns a tangent vector parameterized as
        (Up, M, Vp), as described in the class docstring.
        """
        ZV = self._apply_ambient(Z, X[2])
        UtZV = X[0].T @ ZV
        ZtU = self._apply_ambient_transpose(Z, X[0])

        Up = ZV - X[0] @ UtZV
        M = UtZV
        Vp = ZtU - X[2] @ UtZV.T

        return _TangentVector((Up, M, Vp))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Euclidean part
        Up, M, Vp = self.proj(X, ehess)

        # Curvature part
        s = np.diag(X[1])
        T = self._apply_ambient(egrad, H[2]) / s
        Up += (1 - X[0] @ X[0].T) @ T

        T = self._apply_ambient_transpose(egrad, H[0]) / s
        Vp += (1 - X[2] @ X[2].T) @ T

        return _TangentVector((Up, M, Vp))

    # This retraction is second order, following general results from
    # Absil, Malick, "Projection-like retractions on matrix manifolds",
    # SIAM J. Optim., 22 (2012), pp. 135-158.
    def retr(self, X, Z):
        Qu, Ru = np.linalg.qr(np.hstack((X[0], Z[0])))
        Qv, Rv = np.linalg.qr(np.hstack((X[2], Z[2])))

        # FIXME(nkoep): The identity matrix must be scaled by what Manopt calls
        #               t!
        Id = np.eye(self._k)
        T = np.block([
            [X[1] + Z[1], Id],
            [Id, np.zeros((self._k, self._k))]
        ])

        Ut, St, Vt = np.linalg.svd(Ru @ T @ Rv.T, full_matrices=False)
        Vt = Vt.T

        U = Qu @ Ut[:, :self._k]
        S = np.diag(St[:self._k])
        V = Qv @ Vt[:, :self._k]
        return (U, S, V)

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def rand(self):
        u = self._stiefel_m.rand()
        s = np.diag(np.sort(np.random.rand(self._k))[::-1])
        v = self._stiefel_n.rand()
        return (u, s, v)

    def _tangent(self, X, Z):
        """
        Given Z in tangent vector format, projects the components Up and Vp
        such that they satisfy the tangent space constraints up to numerical
        errors. If Z was indeed a tangent vector at X, this should barely
        affect Z (it would not at all if we had infinite numerical accuracy).
        """
        Up = Z[0] - X[0] @ X[0].T @ Z[0]
        Vp = Z[2] - X[2] @ X[2].T @ Z[2]

        return _TangentVector((Up, Z[1], Vp))

    def randvec(self, X):
        Up = np.random.randn(self._m, self._k)
        M = np.random.randn(self._k, self._k)
        Vp = np.random.randn(self._n, self._k)
        Z = self._tangent(X, (Up, M, Vp))
        norm = self.norm(X, Z)
        return _TangentVector((Z[0] / norm, Z[1] / norm, Z[2] / norm))

    def tangent2ambient(self, X, Z):
        """Transforms a tangent vector Z represented as a structure (Up, M, Vp)
        into a structure with fields (U, S, V) that represents that same
        tangent vector in the ambient space of m x n matrices, as U @ S @ V.T.
        This matrix is equal to X.U @ Z.M @ X.V.T + Z.Up @ X.V.T + X.U @
        Z.Vp.T. The latter is an m x n matrix, which could be too large to
        build explicitly, and this is why we return a low-rank representation
        instead. Note that there are no guarantees on U, S and V other than
        that U @ S @ V.T is the desired matrix. In particular, U and V are not
        (in general) orthonormal and S is not (in general) diagonal.
        (In this implementation, S is identity, but this might change.)
        """
        U = np.hstack((X[0] @ Z[1] + Z[0], X[0]))
        S = np.eye(2 * self._k)
        V = np.hstack((X[2], Z[2]))
        return (U, S, V)

    # Comment from Manopt:
    # New vector transport on June 24, 2014 (as indicated by Bart)
    # Reference: Absil, Mahony, Sepulchre 2008 section 8.1.3:
    # For Riemannian submanifolds of a Euclidean space, it is acceptable to
    # transport simply by orthogonal projection of the tangent vector
    # translated in the ambient space.
    def transp(self, X1, X2, G):
        return self.proj(X2, self.tangent2ambient(X1, G))

    def zerovec(self, X):
        return _TangentVector((np.zeros((self._m, self._k)),
                               np.zeros((self._k, self._k)),
                               np.zeros((self._n, self._k))))


class _TangentVector(tuple, ndarraySequenceMixin):
    def __repr__(self):
        return "{:s}: {}".format(self.__class__.__name__, super().__repr__())

    def to_ambient(self, x):
        Z1 = x[0] @ self[1] @ x[2].T
        Z2 = self[0] @ x[2].T
        Z3 = x[0] @ self[2].T
        return Z1 + Z2 + Z3

    def __add__(self, other):
        return _TangentVector((s + o for (s, o) in zip(self, other)))

    def __sub__(self, other):
        return _TangentVector((s - o for (s, o) in zip(self, other)))

    def __mul__(self, other):
        return _TangentVector((other * s for s in self))

    __rmul__ = __mul__

    def __div__(self, other):
        return _TangentVector((val / other for val in self))

    def __neg__(self):
        return _TangentVector((-val for val in self))
