"""Module containing manifolds of fixed-rank matrices."""

import collections

import numpy as np

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance


class FixedRankEmbedded(EuclideanEmbeddedSubmanifold):
    r"""Manifold of fixed rank matrices.

    Args:
        m: The number of rows of the matrices in the ambient space.
        n: The number of columns of the matrices.
        k: The rank of the matrices on the manifold.

    Notes:
        * The implementation follows the embedded geometry described in
          [Van2013]_.
        * The class is currently not compatible with the
          :class:`pymanopt.optimizers.TrustRegions` optimizer.

    Manifold of ``m x n`` real matrices of fixed rank ``k``.
    For efficiency purposes, Pymanopt does not represent points on this
    manifold explicitly using ``m x n`` matrices, but instead implicitly using
    a truncated singular value decomposition.
    Specifically, a point is represented by a tuple ``(u, s, vt)`` of three
    numpy arrays.
    The arrays ``u``, ``s`` and ``vt`` have shapes ``(m, k)``, ``(k,)`` and
    ``(k, n)``, respectively, and the low rank matrix which they represent can
    be recovered by the matrix product ``u * diag(s) * vt``.

    For example, to optimize over the space of 5 x 4 matrices with rank 3, we

    would need to

    >>> import pymanopt.manifolds
    >>> manifold = pymanopt.manifolds.FixedRankEmbedded(5, 4, 3)

    Then the shapes will be as follows:

    >>> u, s, vt = manifold.rand()
    >>> u.shape
    (5, 3)
    >>> s.shape
    (3,)
    >>> vt.shape
    (3, 4)

    and the full matrix can be recovered using the matrix product
    ``u @ diag(s) @ vt``:

    >>> import numpy as np
    >>> X = u @ np.diag(s) @ vt

    Tangent vectors are represented as a tuple ``(Up, M, Vp)``.
    The matrices ``Up`` (of size ``m x k``) and ``Vp`` (of size ``n x k``) obey
    ``Up' * U = 0 and Vp' * V = 0``.
    The matrix ``M`` (of size ``k x k``) is arbitrary.
    Such a structure corresponds to the
    following tangent vector in the ambient space of ``m x n`` matrices: ``Z =
    U * M * V' + Up * V' + U * Vp'``
    where ``(U, S, V)`` is the current point and ``(Up, M, Vp)`` is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as ``m x n`` matrices.
    If these are low-rank, they may also be represented as structures with
    ``U, S, V`` fields, such that ``Z = U * S * V'``.
    There are no restrictions on what ``U``, ``S`` and ``V`` are, as long as
    their product as indicated yields a real ``m x n`` matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space :math:`\R^(m \times n)` equipped with the usual trace (Frobenius)
    inner product.
    """

    def __init__(self, m: int, n: int, k: int):
        self._m = m
        self._n = n
        self._k = k
        self._stiefel_m = Stiefel(m, k)
        self._stiefel_n = Stiefel(n, k)

        name = f"Embedded manifold of {m}x{n} matrices of rank {k}"
        dimension = (m + n - k) * k
        super().__init__(name, dimension, point_layout=3)

    @property
    def typical_dist(self):
        return self.dim

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.sum(
            np.tensordot(a, b)
            for (a, b) in zip(tangent_vector_a, tangent_vector_b)
        )

    def _apply_ambient(self, vector, matrix):
        """Right-multiply a matrix to a vector in ambient space."""
        if isinstance(vector, (list, tuple)):
            return vector[0] @ vector[1] @ vector[2].T @ matrix
        return vector @ matrix

    def _apply_ambient_transpose(self, vector, matrix):
        """Right-multiply a matrix to transpose of a vector in ambient space."""
        if isinstance(vector, (list, tuple)):
            return vector[2] @ vector[1] @ vector[0].T @ matrix
        return vector.T @ matrix

    def proj(self, point, vector):
        """Project vector to tangent space.

        Note that ``vector`` must either be an m x n matrix from the ambient
        space, or else a tuple (Uz, Sz, Vz), where Uz * Sz * Vz is in the
        ambient space (of low-rank matrices).
        This function then returns a tangent vector parameterized as
        (Up, M, Vp).
        """
        ZV = self._apply_ambient(vector, point[2].T)
        UtZV = point[0].T @ ZV
        ZtU = self._apply_ambient_transpose(vector, point[0])

        Up = ZV - point[0] @ UtZV
        M = UtZV
        Vp = ZtU - point[2].T @ UtZV.T

        return _FixedRankTangentVector(Up, M, Vp)

    def egrad2rgrad(self, point, euclidean_gradient):
        """Convert Euclidean to Riemannian gradient.

        Assuming that the cost function being optimized has been defined
        in terms of the low-rank singular value decomposition, the
        gradient returned by the autodiff backends will have three components
        and will be in the form of a tuple ``euclidean_gradient = (df/dU,
        df/dS, df/dV)``.

        Notes:
            See https://j-towns.github.io/papers/svd-derivative.pdf for a
            detailed explanation of this implementation.
        """
        u, s, vt = point
        du, ds, dvt = euclidean_gradient

        utdu = u.T @ du
        uutdu = u @ utdu
        Up = (du - uutdu) / s

        vtdv = vt @ dvt.T
        vvtdv = vt.T @ vtdv
        Vp = (dvt.T - vvtdv) / s

        identity = np.eye(self._k)
        f = 1 / (s[np.newaxis, :] ** 2 - s[:, np.newaxis] ** 2 + identity)

        M = (
            f * (utdu - utdu.T) * s
            + s[:, np.newaxis] * f * (vtdv - vtdv.T)
            + np.diag(ds)
        )

        return _FixedRankTangentVector(Up, M, Vp)

    # TODO(nkoep): Implement the 'weingarten' method to support the
    #              trust-region optimizer, cf.
    #              https://sites.uclouvain.be/absil/2013-01/Weingarten_07PA_techrep.pdf

    # This retraction is second order, following general results from
    # Absil, Malick, "Projection-like retractions on matrix manifolds",
    # SIAM J. Optim., 22 (2012), pp. 135-158.
    def retr(self, point, tangent_vector):
        u, s, vt = point
        du, ds, dvt = tangent_vector

        Qu, Ru = np.linalg.qr(du)
        Qv, Rv = np.linalg.qr(dvt)
        T = np.vstack(
            (
                np.hstack((np.diag(s) + ds, Rv.T)),
                np.hstack((Ru, np.zeros((self._k, self._k)))),
            )
        )
        # Numpy svd outputs St as a 1d vector, not a matrix.
        Ut, St, Vt = np.linalg.svd(T, full_matrices=False)
        # Transpose because numpy outputs it the wrong way.
        Vt = Vt.T

        U = np.hstack((u, Qu)) @ Ut[:, : self._k]
        S = St[: self._k] + np.spacing(1)
        V = np.hstack((vt.T, Qv)) @ Vt[:, : self._k]
        return _FixedRankPoint(U, S, V.T)

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner(point, tangent_vector, tangent_vector))

    def rand(self):
        u = self._stiefel_m.rand()
        s = np.sort(np.random.rand(self._k))[::-1]
        vt = self._stiefel_n.rand().T
        return _FixedRankPoint(u, s, vt)

    def tangent(self, point, vector):
        """Project components of ``vector`` to tangent space at ``point``.

        Given ``vector`` in tangent vector format, projects its components Up
        and Vp such that they satisfy the tangent space constraints up to
        numerical errors.
        If ``vector`` was indeed a tangent vector at ``point``, this should
        barely affect ``vector``.
        """
        u, _, vt = point
        Up = vector.Up - u @ u.T @ vector.Up
        Vp = vector.Vp - vt.T @ vt @ vector.Vp
        return _FixedRankTangentVector(Up, vector.M, Vp)

    def randvec(self, point):
        Up = np.random.randn(self._m, self._k)
        Vp = np.random.randn(self._n, self._k)
        M = np.random.randn(self._k, self._k)

        tangent_vector = self.tangent(
            point, _FixedRankTangentVector(Up, M, Vp)
        )
        return tangent_vector / self.norm(point, tangent_vector)

    def tangent2ambient(self, point, tangent_vector):
        """Represent tangent vector in ambient space.

        Transforms a tangent vector Z represented as a structure (Up, M, Vp)
        into a structure with fields (U, S, V) that represents that same
        tangent vector in the ambient space of mxn matrices, as U*S*V'.
        This matrix is equal to X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'.
        The latter is an mxn matrix, which could be too large to build
        explicitly, and this is why we return a low-rank representation
        instead.
        Note that there are no guarantees on U, S and V other than that USV' is
        the desired matrix.
        In particular, U and V are not (in general) orthonormal and S is not
        (in general) diagonal.
        Currently, S is identity, but this might change.
        """
        u, _, vt = point
        U = np.hstack((u @ tangent_vector.M + tangent_vector.Up, u))
        S = np.eye(2 * self._k)
        V = np.hstack(([vt.T, tangent_vector.Vp]))
        return U, S, V

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.proj(
            point_b, self.tangent2ambient(point_a, tangent_vector_a)
        )

    def zerovec(self, point):
        return _FixedRankTangentVector(
            np.zeros((self._m, self._k)),
            np.zeros((self._k, self._k)),
            np.zeros((self._n, self._k)),
        )


class _ndarraySequence(ndarraySequenceMixin):
    @return_as_class_instance
    def __mul__(self, other):
        return [other * s for s in self]

    __rmul__ = __mul__

    @return_as_class_instance
    def __truediv__(self, other):
        return [val / other for val in self]

    @return_as_class_instance
    def __neg__(self):
        return [-val for val in self]


class _FixedRankPoint(
    _ndarraySequence,
    collections.namedtuple(
        "_FixedRankPointTuple", field_names=("u", "s", "vt")
    ),
):
    pass


class _FixedRankTangentVector(
    _ndarraySequence,
    collections.namedtuple(
        "_FixedRankTangentVectorTuple", field_names=("Up", "M", "Vp")
    ),
):
    @return_as_class_instance
    def __add__(self, other):
        return [s + o for (s, o) in zip(self, other)]

    @return_as_class_instance
    def __sub__(self, other):
        return [s - o for (s, o) in zip(self, other)]
