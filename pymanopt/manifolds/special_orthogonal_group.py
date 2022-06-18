import numpy as np
import scipy.special

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import (
    multiexpm,
    multilogm,
    multiqr,
    multiskew,
    multisym,
    multitransp,
)


class SpecialOrthogonalGroup(RiemannianSubmanifold):
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

    Args:
        n: The dimension of the space that elements of the group act on.
        k: The number of elements in the product of groups.
        retraction: The type of retraction to use.
            Possible choices are ``qr`` and ``polar``.

    Note:
        The default SVD-based retraction is only a first-order approximation of
        the exponential map.
        Use of a second-order retraction can be enabled by instantiating the
        class with ``SpecialOrthogonalGroup(n, k=k, retraction="polar")``.

        The procedure to generate random rotation matrices sampled uniformly
        from the Haar measure is detailed in [Mez2006]_.
    """

    def __init__(self, n: int, *, k: int = 1, retraction: str = "qr"):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Special orthogonal group SO({n})"
        elif k > 1:
            name = f"Sphecial orthogonal group SO({n})^{k}"
        else:
            raise ValueError("k must be an integer no less than 1.")
        dimension = int(k * scipy.special.comb(n, 2))
        super().__init__(name, dimension)

        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    @property
    def typical_dist(self):
        return np.pi * np.sqrt(self._n * self._k)

    def dist(self, point_a, point_b):
        return self.norm(point_a, self.log(point_a, point_b))

    def projection(self, point, vector):
        return multiskew(multitransp(point) @ vector)

    def to_tangent_space(self, point, vector):
        return multiskew(vector)

    def embedding(self, point, tangent_vector):
        return point @ tangent_vector

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        Xt = multitransp(point)
        Xtegrad = Xt @ euclidean_gradient
        symXtegrad = multisym(Xtegrad)
        Xtehess = Xt @ euclidean_hessian
        return multiskew(Xtehess - tangent_vector @ symXtegrad)

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        Y = point + point @ tangent_vector
        q, _ = multiqr(Y)
        return q

    def _retraction_polar(self, point, tangent_vector):
        Y = point + point @ tangent_vector
        u, _, vt = np.linalg.svd(Y)
        return u @ vt

    def exp(self, point, tangent_vector):
        return point @ multiexpm(tangent_vector)

    def log(self, point_a, point_b):
        return multiskew(multilogm(multitransp(point_a) @ point_b))

    def random_point(self):
        n, k = self._n, self._k
        if n == 1:
            point = np.ones((k, 1, 1))
        else:
            point, _ = multiqr(np.random.normal(size=(k, n, n)))
            # Swap the first two columns of matrices where det(point) < 0 to
            # flip the sign of their determinants.
            negative_det, *_ = np.where(np.linalg.det(point) < 0)
            slice_ = np.arange(point.shape[1])
            point[np.ix_(negative_det, slice_, [0, 1])] = point[
                np.ix_(negative_det, slice_, [1, 0])
            ]
        if k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        n, k = self._n, self._k
        inds = np.triu_indices(n, 1)
        vector = np.zeros((k, n, n))
        for i in range(k):
            vector[i][inds] = np.random.normal(size=int(n * (n - 1) / 2))
        vector = vector - multitransp(vector)
        if k == 1:
            vector = vector[0]
        return vector / np.sqrt(np.tensordot(vector, vector, axes=vector.ndim))

    def zero_vector(self, point):
        zero = np.zeros((self._k, self._n, self._n))
        if self._k == 1:
            return zero[0]
        return zero

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pair_mean(self, point_a, point_b):
        V = self.log(point_a, point_b)
        return self.exp(point_a, 0.5 * V)
