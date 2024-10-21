from typing import Optional

import scipy.special

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.numerics import NumericsBackend
from pymanopt.tools import extend_docstring


class _UnitaryBase(RiemannianSubmanifold):
    _n: int
    _k: int

    def __init__(
        self,
        name,
        dimension,
        retraction,
        backend: Optional[NumericsBackend] = None,
    ):
        super().__init__(name, dimension, backend=backend)

        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self.backend.tensordot(
            self.backend.conjugate(tangent_vector_a),
            tangent_vector_b,
            axes=tangent_vector_a.ndim,
        )

    def norm(self, point, tangent_vector):
        return self.backend.linalg_norm(tangent_vector)

    @property
    def typical_dist(self):
        return self.backend.pi * self.backend.sqrt(self._n * self._k)

    def dist(self, point_a, point_b):
        return self.norm(point_a, self.log(point_a, point_b))

    def projection(self, point, vector):
        return self.backend.skew(
            self.backend.conjugate_transpose(point) @ vector
        )

    def to_tangent_space(self, point, vector):
        return self.backend.skewh(vector)

    def embedding(self, point, tangent_vector):
        return point @ tangent_vector

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        Xt = self.backend.conjugate(point)
        Xtegrad = Xt @ euclidean_gradient
        symXtegrad = self.backend.herm(Xtegrad)
        Xtehess = Xt @ euclidean_hessian
        return self.backend.skewh(Xtehess - tangent_vector @ symXtegrad)

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        Y = point + point @ tangent_vector
        q, _ = self.backend.linalg_qr(Y)
        return q

    def _retraction_polar(self, point, tangent_vector):
        Y = point + point @ tangent_vector
        u, _, vt = self.backend.linalg_svd(Y)
        return u @ vt

    def exp(self, point, tangent_vector):
        return point @ self.backend.linalg_expm(tangent_vector)

    def log(self, point_a, point_b):
        return self.backend.skewh(
            self.backend.linalg_logm(
                self.backend.conjugate_transpose(point_a) @ point_b
            )
        )

    def zero_vector(self, point):
        zero = self.backend.zeros((self._k, self._n, self._n))
        if self._k == 1:
            return zero[0]
        return zero

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pair_mean(self, point_a, point_b):
        return self.exp(point_a, self.log(point_a, point_b) / 2)

    def _random_skew_symmetric_matrix(self, n, k):
        if n == 1:
            return self.backend.zeros((k, 1, 1))
        vector = self._random_upper_triangular_matrix(n, k)
        return vector - self.backend.transpose(vector)

    def _random_symmetric_matrix(self, n, k):
        if n == 1:
            return self.backend.random_normal(size=(k, 1, 1))
        vector = self._random_upper_triangular_matrix(n, k)
        vector = vector + self.backend.transpose(vector)
        # The diagonal elements get scaled by a factor of 2 by the previous
        # operation so re-draw them so every entry of the returned matrix follows a
        # standard normal distribution.
        indices = self.backend.arange(n)
        vector[:, indices, indices] = self.backend.random_normal(size=(k, n))
        return vector

    def _random_upper_triangular_matrix(self, n, k):
        if n < 2:
            raise ValueError("Matrix dimension cannot be less than 2")
        indices = self.backend.triu_indices(n, 1)
        vector = self.backend.zeros((k, n, n))
        vector[(slice(None), *indices)] = self.backend.random_normal(
            size=(k, n * (n - 1) // 2)
        )
        return vector


DOCSTRING_NOTE = """
    Args:
        n: The dimension of the space that elements of the group act on.
        k: The number of elements in the product of groups.
        retraction: The type of retraction to use.
            Possible choices are ``qr`` and ``polar``.

    Note:
        The default QR-based retraction is only a first-order approximation of
        the exponential map.
        Use of an SVD-based second-order retraction can be enabled by setting
        the ``retraction`` argument to "polar".

        The procedure to generate random points on the manifold sampled
        uniformly from the Haar measure is detailed in [Mez2006]_.
"""


@extend_docstring(DOCSTRING_NOTE)
class SpecialOrthogonalGroup(_UnitaryBase):
    r"""The (product) manifold of rotation matrices.

    The special orthgonal group :math:`\SO(n)`.
    Points on the manifold are matrices :math:`\vmQ \in \R^{n
    \times n}` such that each matrix is orthogonal with determinant 1, i.e.,
    :math:`\transp{\vmQ}\vmQ = \vmQ\transp{\vmQ} = \Id_n` and :math:`\det(\vmQ)
    = 1`.
    For ``k > 1``, the class can be used to optimize over the product manifold
    of rotation matrices :math:`\SO(n)^k`.
    In that case points on the manifold are represented as arrays of shape
    ``(k, n, n)``.

    The metric is the usual Euclidean one inherited from the embedding space
    :math:`(\R^{n \times n})^k`.
    As such :math:`\SO(n)^k` forms a Riemannian submanifold.

    The tangent space :math:`\tangent{\vmQ}\SO(n)` at a point :math:`\vmQ` is
    given by :math:`\tangent{\vmQ}\SO(n) = \set{\vmQ \vmOmega \in \R^{n \times
    n} \mid \vmOmega = -\transp{\vmOmega}} = \vmQ \Skew(n)`, where
    :math:`\Skew(n)` denotes the set of skew-symmetric matrices.
    This corresponds to the Lie algebra of :math:`\SO(n)`, a fact which is used
    here to conveniently represent tangent vectors numerically by their
    skew-symmetric factor.
    The method :meth:`embedding` can be used to transform a tangent vector from
    its Lie algebra representation to the embedding space representation.
    """

    def __init__(
        self,
        n: int,
        *,
        k: int = 1,
        retraction: str = "qr",
        backend: Optional[NumericsBackend] = None,
    ):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Special orthogonal group SO({n})"
        elif k > 1:
            name = f"Sphecial orthogonal group SO({n})^{k}"
        else:
            raise ValueError("k must be an integer no less than 1.")
        dimension = int(k * scipy.special.comb(n, 2))
        super().__init__(name, dimension, retraction, backend=backend)

    def random_point(self):
        n, k = self._n, self._k
        if n == 1:
            point = self.backend.ones((k, 1, 1))
        else:
            point, _ = self.backend.linalg_qr(
                self.backend.random_normal(size=(k, n, n))
            )
            # Swap the first two columns of matrices where det(point) < 0 to
            # flip the sign of their determinants.
            negative_det, *_ = self.backend.where(
                self.backend.linalg_det(point) < 0
            )
            negative_det = self.backend.expand_dims(negative_det, (-2, -1))
            point[negative_det, :, [0, 1]] = point[negative_det, :, [1, 0]]
        if k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        vector = self._random_skew_symmetric_matrix(self._n, self._k)
        if self._k == 1:
            vector = vector[0]
        return vector / self.norm(point, vector)


@extend_docstring(DOCSTRING_NOTE)
class UnitaryGroup(_UnitaryBase):
    r"""The (product) manifold of unitary matrices (i.e., the unitary group).

    The unitary group :math:`\U(n)`.
    Points on the manifold are matrices :math:`\vmX \in \C^{n
    \times n}` such that each matrix is unitary, i.e.,
    :math:`\transp{\conj{\vmX}}\vmX = \adj{\vmX}\vmX = \Id_n`.
    For ``k > 1``, the class represents the product manifold
    of unitary matrices :math:`\U(n)^k`.
    In that case points on the manifold are represented as arrays of shape
    ``(k, n, n)``.

    The metric is the usual Euclidean one inherited from the embedding space
    :math:`(\C^{n \times n})^k`, i.e., :math:`\inner{\vmA}{\vmB} =
    \Re\tr(\adj{\vmA}\vmB)`.
    As such :math:`\U(n)^k` forms a Riemannian submanifold.

    The tangent space :math:`\tangent{\vmX}\U(n)` at a point :math:`\vmX` is
    given by :math:`\tangent{\vmX}\U(n) = \set{\vmX \vmOmega \in \C^{n \times
    n} \mid \vmOmega = -\adj{\vmOmega}} = \vmX \adj{\Skew}(n)`, where
    :math:`\adj{\Skew}(n)` denotes the set of skew-Hermitian matrices.
    This corresponds to the Lie algebra of :math:`\U(n)`, a fact which is used
    here to conveniently represent tangent vectors numerically by their
    skew-Hermitian factor.
    The method :meth:`embedding` can be used to convert a tangent vector from
    its Lie algebra representation to the embedding space representation.
    """

    def __init__(
        self,
        n: int,
        *,
        k: int = 1,
        retraction: str = "qr",
        backend: Optional[NumericsBackend] = None,
    ):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Unitary group U({n})"
        elif k > 1:
            name = f"Unitary group U({n})^{k}"
        else:
            raise ValueError("k must be an integer no less than 1.")
        dimension = int(k * n**2)
        super().__init__(name, dimension, retraction, backend=backend)

    def random_point(self):
        n, k = self._n, self._k
        if n == 1:
            point = self.backend.ones((k, 1, 1)) + 1j * self.backend.ones(
                (k, 1, 1)
            )
            point /= self.backend.abs(point)
        else:
            point, _ = self.backend.linalg_qr(
                self.backend.random_normal(size=(k, n, n))
                + 1j * self.backend.random_normal(size=(k, n, n))
            )
        if k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        n, k = self._n, self._k
        vector = (
            self._random_skew_symmetric_matrix(n, k)
            + 1j * self._random_symmetric_matrix(n, k)
        ) / self.backend.sqrt(2)
        if k == 1:
            vector = vector[0]
        return vector / self.norm(point, vector)
