import collections

import numpy as np

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance


class FixedRankEmbedded(RiemannianSubmanifold):
    r"""Manifold of fixed rank matrices.

    Args:
        m: The number of rows of matrices in the ambient space.
        n: The number of columns of matrices in the ambient space.
        k: The rank of matrices.

    The manifold of ``m x n`` real matrices of fixed rank ``k``.
    For efficiency purposes, points on the manifold are represented with a
    truncated singular value decomposition instead of full matrices of size ``m
    x n``.
    Specifically, a point is represented as a tuple ``(u, s, vt)`` of three
    arrays.
    The arrays ``u``, ``s`` and ``vt`` have shapes ``(m, k)``, ``(k,)`` and
    ``(k, n)``, respectively, and the rank ``k`` matrix which they represent
    can be recovered by the product ``u @ np.diag(s) @ vt``.

    Vectors ``Z`` in the ambient space are best represented as arrays of shape
    ``(m, n)``.
    If these are low-rank, they may also be represented as tuples of arrays
    ``(U, S, V)`` such that ``Z = U @ S @ V.T``.
    There are no restrictions on what ``U``, ``S`` and ``V`` are, as long as
    their product as indicated yields a real ``m x n`` matrix.

    Tangent vectors are represented as tuples of the form ``(Up, M, Vp)``.
    The matrices ``Up`` (of size ``m x k``) and ``Vp`` (of size ``n x k``) obey
    the conditions ``np.allclose(Up.T @ U, 0)`` and ``np.allclose(Vp.T @ V,
    0)``.
    The matrix ``M`` (of size ``k x k``) is arbitrary.
    Such a structure corresponds to the tangent vector ``Z = u @ M @ vt + Up @
    vt + u * Vp.T`` in the ambient space of ``m x n`` matrices at a point ``(u,
    s, vt)``.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space :math:`\R^{m \times n}` equipped with the usual trace inner product.

    Note:
        * The implementation follows the embedded geometry described in
          [Van2013]_.
        * The class is currently not compatible with the
          :class:`pymanopt.optimizers.trust_regions.TrustRegions` optimizer.
        * Details on the implementation of
          :meth:`euclidean_to_riemannian_gradient` can be found at
          https://j-towns.github.io/papers/svd-derivative.pdf.
        * The second-order retraction follows results presented in [AM2012]_.
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

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.sum(
            [
                np.tensordot(a, b)
                for (a, b) in zip(tangent_vector_a, tangent_vector_b)
            ]
        )

    def _apply_ambient(self, vector, matrix):
        if isinstance(vector, (list, tuple)):
            return vector[0] @ vector[1] @ vector[2].T @ matrix
        return vector @ matrix

    def _apply_ambient_transpose(self, vector, matrix):
        if isinstance(vector, (list, tuple)):
            return vector[2] @ vector[1] @ vector[0].T @ matrix
        return vector.T @ matrix

    def projection(self, point, vector):
        """Project vector in the ambient space to the tangent space.

        Args:
            point: A point on the manifold.
            vector: A vector in the ambient space.

        Returns:
            A tangent vector parameterized as a ``(Up, M, Vp)``.

        Note:
            The argument ``vector`` must either be an array of shape ``(m, n)``
            in the ambient space, or else a tuple ``(U, S, V)`` where ``U @ S @
            V`` is in the ambient space (of low-rank matrices).
        """
        ZV = self._apply_ambient(vector, point[2].T)
        UtZV = point[0].T @ ZV
        ZtU = self._apply_ambient_transpose(vector, point[0])

        Up = ZV - point[0] @ UtZV
        M = UtZV
        Vp = ZtU - point[2].T @ UtZV.T

        return _FixedRankTangentVector(Up, M, Vp)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
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

    def retraction(self, point, tangent_vector):
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

        U = np.hstack((u, Qu)) @ Ut[:, : self._k]
        S = St[: self._k] + np.spacing(1)
        V = np.hstack((vt.T, Qv)) @ Vt.T[:, : self._k]
        return _FixedRankPoint(U, S, V.T)

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        u = self._stiefel_m.random_point()
        s = np.sort(np.random.uniform(size=self._k))[::-1]
        vt = self._stiefel_n.random_point().T
        return _FixedRankPoint(u, s, vt)

    def to_tangent_space(self, point, vector):
        u, _, vt = point
        Up = vector.Up - u @ u.T @ vector.Up
        Vp = vector.Vp - vt.T @ vt @ vector.Vp
        return _FixedRankTangentVector(Up, vector.M, Vp)

    def random_tangent_vector(self, point):
        Up = np.random.normal(size=(self._m, self._k))
        Vp = np.random.normal(size=(self._n, self._k))
        M = np.random.normal(size=(self._k, self._k))

        tangent_vector = self.to_tangent_space(
            point, _FixedRankTangentVector(Up, M, Vp)
        )
        return tangent_vector / self.norm(point, tangent_vector)

    def embedding(self, point, tangent_vector):
        u, _, vt = point
        U = np.hstack((u @ tangent_vector.M + tangent_vector.Up, u))
        S = np.eye(2 * self._k)
        V = np.hstack(([vt.T, tangent_vector.Vp]))
        return U, S, V

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(
            point_b, self.embedding(point_a, tangent_vector_a)
        )

    def zero_vector(self, point):
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
