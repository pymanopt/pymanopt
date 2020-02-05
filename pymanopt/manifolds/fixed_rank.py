"""
Module containing manifolds of fixed rank matrices.
"""
import collections

import numpy as np

from pymanopt.manifolds.manifold import (EuclideanEmbeddedSubmanifold,
                                         RetrAsExpMixin)
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.tools import ndarraySequenceMixin


class FixedRankEmbedded(EuclideanEmbeddedSubmanifold, RetrAsExpMixin):
    """Manifold of m-by-n real matrices of fixed rank k.

    This follows the embedded geometry described in Bart Vandereycken's 2013
    paper: "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    For efficiency purposes, Pymanopt does not represent points on this
    manifold explicitly using m x n matrices, but instead implicitly using
    a truncated singular value decomposition. Specifically, a point is
    represented by a tuple (U, S, V) of three numpy arrays. The arrays U,
    S and V have shapes m x k, k x k and n x k, respectively, and the low
    rank matrix which they represent can be recovered by the matrix product
    U @ S @ V.T.

    For example, to optimize over the space of 5 by 4 matrices with rank 3,
    we would need to
    >>> from pymanopt.manifolds import FixedRankEmbedded
    >>> manifold = FixedRankEmbedded(5, 4, 3)

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
    >>> X = x[0] @ x[1] @ x[2].T

    Tangent vectors are represented as a tuple (Up, M, Vp) where the arrays
    are of size m x k, k x k and n x k, respectively. The matrices Up and Vp
    obey
        Up.T @ U == np.zeros((k, k))
    and
        Vp.T @ V == np.zeros((k, k)).
    The matrix M is arbitrary. Such a structure corresponds to the following
    tangent vector in the ambient space of m x n matrices:
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
    """

    class _TangentVector(
            collections.namedtuple("_Triple", field_names=("Up", "M", "Vp")),
            ndarraySequenceMixin):
        def __repr__(self):
            return "{:s}: {}".format(
                self.__class__.__name__, super().__repr__())

        def __add__(self, other):
            return self.__class__(*[s + o for (s, o) in zip(self, other)])

        def __sub__(self, other):
            return self.__class__(*[s - o for (s, o) in zip(self, other)])

        def __mul__(self, other):
            return self.__class__(*[s * other for s in self])

        __rmul__ = __mul__

        def __truediv__(self, other):
            return self.__class__(*[s / other for s in self])

        def __neg__(self):
            return self.__class__(*[-s for s in self])

    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k
        self._stiefel_m = Stiefel(m, k)
        self._stiefel_n = Stiefel(n, k)

        name = ("Manifold of ({m} x {n}) matrices of rank {k}".format(
            m=m, n=n, k=k))
        dimension = (m + n - k) * k
        super().__init__(name, dimension, point_layout=3)

    @property
    def typicaldist(self):
        return self.dim

    def inner(self, X, G, H):
        return np.sum(np.tensordot(a, b) for (a, b) in zip(G, H))

    def norm(self, X, D):
        return np.sqrt(self.inner(X, D, D))

    def tangent(self, X, Z):
        """Tangentializes a tangent vector ``Z`` at a point ``X``.

        Given ``Z`` in tangent vector format ``(Up, M, Vp)``, the method
        projects the components ``Up`` and ``Vp`` such that they satisfy the
        tangent space constraints up to numerical errors. If ``Z`` was indeed a
        tangent vector at ``X``, this should barely affect ``Z`` (it would not
        at all if we had infinite numerical accuracy).
        """
        Up = Z.Up - X[0] @ X[0].T @ Z.Up
        Vp = Z.Vp - X[2] @ X[2].T @ Z.Vp
        return self._TangentVector(Up, Z.M, Vp)

    def _apply_ambient(self, Z, W):
        """Right-multiplies an ambient vector by a matrix.

        Given an ambient vector ``Z`` represented as an array of shape ``(m,
        n)`` or a tuple ``(U, S, V)``, the method left-multiplies the vector to
        the matrix ``W``.
        """
        if isinstance(Z, (list, tuple)):
            return Z[0] @ Z[1] @ Z[2].T @ W
        return Z @ W

    def _apply_ambient_transpose(self, Z, W):
        """Right-multiplies the transpose of an ambient vector by a matrix.

        This is the same as ``_apply_ambient``, but the vector ``Z`` is
        transposed before right-multiplying it with ``W``.
        """
        if isinstance(Z, (list, tuple)):
            return Z[2] @ Z[1].T @ Z[0].T @ W
        return Z.T @ W

    def proj(self, X, Z):
        ZV = self._apply_ambient(Z, X[2])
        UtZV = X[0].T @ ZV
        ZtU = self._apply_ambient_transpose(Z, X[0])

        Up = ZV - X[0] @ UtZV
        M = UtZV
        Vp = ZtU - X[2] @ UtZV.T
        return self._TangentVector(Up, M, Vp)

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Euclidean part
        rhess = self.proj(X, ehess)

        # Curvature part
        s = np.diag(X[1])
        T = self._apply_ambient(egrad, H.Vp) / s
        Up = rhess.Up + T - X[0] @ X[0].T @ T
        T = self._apply_ambient_transpose(egrad, H.Up) / s
        Vp = rhess.Vp + T - X[2] @ X[2].T @ T

        return self._TangentVector(Up, rhess.M, Vp)

    def tangent2ambient(self, X, Z):
        """Represent a tangent vector in the ambient space.

        Transforms a tangent vector ``Z`` represented as a tuple ``(Up, M,
        Vp)`` to a tuple ``(U, S, V)`` that represents the same tangent vector
        in the ambient space of m x n matrices as U @ S @ V.T. The matrix is
        equal to
          X[0] @ Z[1] @ X[2].T + Z[0] @ X.[2].T + X[0] @ Z[2].T.
        The latter is an m x n matrix, which could be too large to build
        explicitly, and this is why we return a low-rank representation
        instead. Note that there are no guarantees on ``U``, ``S`` and ``V``
        other than that U @ S @ V.T is the desired matrix. In particular, ``U``
        and ``V`` are not (in general) orthonormal and ``S`` is not (in
        general) diagonal.
        """
        U = np.hstack((X[0] @ Z.M + Z.Up, X[0]))
        S = np.eye(2 * self._k)
        V = np.hstack((X[2], Z.Vp))
        return (U, S, V)

    def retr(self, X, Z):
        """
        Notes
        -----
        This retraction is second-order, following general results from [1]_.

        References
        ----------
        .. [1] Absil, Malick, "Projection-like retractions on matrix
           manifolds", SIAM J. Optim., 22 (2012), pp. 135-158.
        """
        Qu, Ru = np.linalg.qr(Z.Up)
        Qv, Rv = np.linalg.qr(Z.Vp)

        k = self._k
        T = np.block([
            [X[1] + Z.M, Rv.T],
            [Ru, np.zeros((k, k))]
        ])
        Ut, St, Vt = np.linalg.svd(T, full_matrices=False)
        Vt = Vt.T

        U = np.hstack((X[0], Qu)) @ Ut[:, :k]
        S = np.diag(St[:k] + np.spacing(1))
        V = np.hstack((X[2], Qv)) @ Vt[:, :k]
        return (U, S, V)

    def rand(self):
        U = self._stiefel_m.rand()
        S = np.diag(np.sort(np.random.randn(self._k))[::-1])
        V = self._stiefel_n.rand()
        return (U, S, V)

    def randvec(self, X):
        m, n, k = self._m, self._n, self._k
        randn = np.random.randn
        T = self._TangentVector(randn(m, k), randn(k, k), randn(n, k))
        Z = self.tangent(X, T)
        return Z / self.norm(X, Z)

    def zerovec(self, X):
        m, n, k = self._m, self._n, self._k
        zeros = np.zeros
        return self._TangentVector(zeros((m, k)), zeros((k, k)), zeros((n, k)))

    def transp(self, X1, X2, Z):
        return self.proj(X2, self.tangent2ambient(X1, Z))
