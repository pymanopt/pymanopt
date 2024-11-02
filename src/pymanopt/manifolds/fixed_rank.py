from collections import namedtuple

from pymanopt.backends import Backend, DummyBackendSingleton
from pymanopt.manifolds.manifold import Manifold, RiemannianSubmanifold
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.tools import ArraySequenceMixin, return_as_class_instance


class _ArrayNamedTupleMixin(ArraySequenceMixin):
    @return_as_class_instance(unpack=True)
    def __neg__(self):
        return [-x for x in self]  # type: ignore

    @return_as_class_instance(unpack=True)
    def __add__(self, other):
        return [x + y for x, y in zip(self, other)]  # type: ignore

    @return_as_class_instance(unpack=True)
    def __sub__(self, other):
        return [x - y for x, y in zip(self, other)]  # type: ignore

    @return_as_class_instance(unpack=True)
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return [x * y for x, y in zip(self, other)]  # type: ignore
        return [other * x for x in self]  # type: ignore

    __rmul__ = __mul__

    @return_as_class_instance(unpack=True)
    def __truediv__(self, other):
        return [x / y for x, y in zip(self, other)]  # type: ignore


class _FixedRankPoint(
    _ArrayNamedTupleMixin, namedtuple("_FixedRankPointTuple", ("u", "s", "vt"))
):
    pass


class _FixedRankTangentVector(
    _ArrayNamedTupleMixin,
    namedtuple("_FixedRankTangentVectorTuple", ("Up", "M", "Vp")),
):
    pass


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
    vt + u @ Vp.T`` in the ambient space of ``m x n`` matrices at a point ``(u,
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

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        backend: Backend = DummyBackendSingleton,
    ):
        self._m = m
        self._n = n
        self._k = k
        self._stiefel_m = Stiefel(m, k, backend=backend)
        self._stiefel_n = Stiefel(n, k, backend=backend)

        name = f"Embedded manifold of {m}x{n} matrices of rank {k}"
        dimension = (m + n - k) * k
        super().__init__(name, dimension, point_layout=3, backend=backend)

    @Manifold.backend.setter
    def backend(self, backend: Backend):
        self._backend = backend
        self._stiefel_m.backend = backend
        self._stiefel_n.backend = backend

    def set_backend_with_default_dtype(self, backend_type: type):
        super().set_backend_with_default_dtype(backend_type)
        self._stiefel_m.set_backend_with_default_dtype(backend_type)
        self._stiefel_n.set_backend_with_default_dtype(backend_type)

    @property
    def typical_dist(self):
        return self.dim

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return sum(
            [
                self.backend.tensordot(a, b)
                for (a, b) in zip(tangent_vector_a, tangent_vector_b)
            ]
        )

    def _apply_ambient(self, vector, matrix):
        if isinstance(vector, (list, tuple)):
            return (
                vector[0]
                @ vector[1]
                @ self.backend.transpose(vector[2])
                @ matrix
            )
        return vector @ matrix

    def _apply_ambient_transpose(self, vector, matrix):
        if isinstance(vector, (list, tuple)):
            return (
                vector[2]
                @ vector[1]
                @ self.backend.transpose(vector[0])
                @ matrix
            )
        return self.backend.transpose(vector) @ matrix

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
        bk = self.backend
        if isinstance(vector, (list, tuple)):
            vector = vector[0] @ vector[1] @ bk.transpose(vector[2])
        # ZV = self._apply_ambient(vector, point[2].T)
        ZV = vector @ bk.transpose(point[2])
        UtZV = bk.transpose(point[0]) @ ZV
        # ZtU = self._apply_ambient_transpose(vector, point[0])
        ZtU = bk.transpose(vector) @ point[0]

        Up = ZV - point[0] @ UtZV
        M = UtZV
        Vp = ZtU - bk.transpose(point[2]) @ bk.transpose(UtZV)

        return _FixedRankTangentVector(Up, M, Vp)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        bk = self.backend
        u, s, vt = point
        du, ds, dvt = euclidean_gradient

        utdu = bk.transpose(u) @ du
        uutdu = u @ utdu
        Up = (du - uutdu) / s

        vtdv = vt @ bk.transpose(dvt)
        vvtdv = bk.transpose(vt) @ vtdv
        Vp = (bk.transpose(dvt) - vvtdv) / s

        identity = self.backend.eye(self._k)
        f = 1 / (
            s[self.backend.newaxis, :] ** 2
            - s[:, self.backend.newaxis] ** 2
            + identity
        )

        M = (
            f * (utdu - bk.transpose(utdu)) * s
            + s[:, self.backend.newaxis] * f * (vtdv - bk.transpose(vtdv))
            + self.backend.diag(ds)
        )

        return _FixedRankTangentVector(Up, M, Vp)

    def retraction(self, point, tangent_vector):
        bk = self.backend
        u, s, vt = point
        du, ds, dvt = tangent_vector

        Qu, Ru = bk.linalg_qr(du)
        Qv, Rv = bk.linalg_qr(dvt)
        T = bk.vstack(
            (
                bk.hstack((bk.diag(s) + ds, bk.transpose(Rv))),
                bk.hstack((Ru, bk.zeros((self._k, self._k)))),
            )
        )
        # Numpy svd outputs St as a 1d vector, not a matrix.
        Ut, St, Vt = bk.linalg_svd(T, full_matrices=False)

        U = bk.hstack((u, Qu)) @ Ut[:, : self._k]
        S = St[: self._k] + bk.eps()
        V = bk.hstack((bk.transpose(vt), Qv)) @ bk.transpose(Vt)[:, : self._k]
        return _FixedRankPoint(U, S, bk.transpose(V))

    def norm(self, point, tangent_vector):
        return self.backend.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        bk = self.backend
        u = self._stiefel_m.random_point()
        s = bk.sort(bk.random_uniform(size=self._k), descending=True)
        vt = bk.transpose(self._stiefel_n.random_point())
        return _FixedRankPoint(u, s, vt)

    def to_tangent_space(self, point, vector):
        bk = self.backend
        u, _, vt = point
        Up = vector.Up - u @ bk.transpose(u) @ vector.Up
        Vp = vector.Vp - bk.transpose(vt) @ vt @ vector.Vp
        return _FixedRankTangentVector(Up, vector.M, Vp)

    def random_tangent_vector(self, point):
        bk = self.backend
        Up = bk.random_normal(size=(self._m, self._k))
        Vp = bk.random_normal(size=(self._n, self._k))
        M = bk.random_normal(size=(self._k, self._k))

        tangent_vector = self.to_tangent_space(
            point, _FixedRankTangentVector(Up, M, Vp)
        )
        return tangent_vector / self.norm(point, tangent_vector)

    def embedding(self, point, tangent_vector):
        bk = self.backend
        u, _, vt = point
        U = bk.hstack((u @ tangent_vector.M + tangent_vector.Up, u))
        S = bk.eye(2 * self._k)
        V = bk.hstack(([bk.transpose(vt), tangent_vector.Vp]))
        return U, S, V

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(
            point_b, self.embedding(point_a, tangent_vector_a)
        )

    def zero_vector(self, point):
        bk = self.backend
        return _FixedRankTangentVector(
            bk.zeros((self._m, self._k)),
            bk.zeros((self._k, self._k)),
            bk.zeros((self._n, self._k)),
        )
