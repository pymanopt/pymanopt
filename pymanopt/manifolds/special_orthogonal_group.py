import numpy as np
from scipy.linalg import expm, logm
from scipy.special import comb

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import multiprod, multiskew, multisym, multitransp


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

    Tangent vectors are represented in the Lie algebra of skew-symmetric
    matrices of the same shape as points on the manifold.
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
        dimension = int(k * comb(n, 2))
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
        return multiskew(multiprod(multitransp(point), vector))

    def to_tangent_space(self, point, vector):
        return multiskew(vector)

    def embedding(self, point, tangent_vector):
        return multiprod(point, tangent_vector)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        Xt = multitransp(point)
        Xtegrad = multiprod(Xt, euclidean_gradient)
        symXtegrad = multisym(Xtegrad)
        Xtehess = multiprod(Xt, euclidean_hessian)
        return multiskew(Xtehess - multiprod(tangent_vector, symXtegrad))

    def retraction(self, point, tangent_vector):
        return self._retraction(point, tangent_vector)

    def _retraction_qr(self, point, tangent_vector):
        def retri(array):
            q, r = np.linalg.qr(array)
            return q @ np.diag(np.sign(np.sign(np.diag(r)) + 0.5))

        Y = point + multiprod(point, tangent_vector)
        if self._k == 1:
            return retri(Y)

        for i in range(self._k):
            Y[i] = retri(Y[i])
        return Y

    def _retraction_polar(self, point, tangent_vector):
        Y = point + multiprod(point, tangent_vector)
        u, _, vt = np.linalg.svd(Y)
        return multiprod(u, vt)

    def exp(self, point, tangent_vector):
        tv = np.copy(tangent_vector)
        if self._k == 1:
            return multiprod(point, expm(tv))

        for i in range(self._k):
            tv[i] = expm(tv[i])
        return multiprod(point, tv)

    def log(self, point_a, point_b):
        U = multiprod(multitransp(point_a), point_b)
        if self._k == 1:
            return multiskew(np.real(logm(U)))

        for i in range(self._k):
            U[i] = np.real(logm(U[i]))
        return multiskew(U)

    def random_point(self):
        n, k = self._n, self._k
        if n == 1:
            R = np.ones((k, 1, 1))
        else:
            R = np.zeros((k, n, n))
            for i in range(k):
                # Generated as such, Q is uniformly distributed over O(n), the
                # group of orthogonal n-by-n matrices.
                A = np.random.normal(size=(n, n))
                Q, RR = np.linalg.qr(A)
                Q = Q @ np.diag(np.sign(np.diag(RR)))

                # If Q is in O(n) but not in SO(n), we permute the two first
                # columns of Q such that det(new Q) = -det(Q), hence the new Q
                # will be in SO(n), uniformly distributed.
                if np.linalg.det(Q) < 0:
                    Q[:, [0, 1]] = Q[:, [1, 0]]
                R[i] = Q

        if k == 1:
            return R.reshape(n, n)
        return R

    def random_tangent_vector(self, point):
        n, k = self._n, self._k
        idxs = np.triu_indices(n, 1)
        vector = np.zeros((k, n, n))
        for i in range(k):
            vector[i][idxs] = np.random.normal(size=int(n * (n - 1) / 2))
            vector = vector - multitransp(vector)
        if k == 1:
            vector = vector.reshape(n, n)
        return vector / np.sqrt(np.tensordot(vector, vector, axes=vector.ndim))

    def zero_vector(self, point):
        if self._k == 1:
            return np.zeros((self._n, self._n))
        return np.zeros((self._k, self._n, self._n))

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pair_mean(self, point_a, point_b):
        V = self.log(point_a, point_b)
        return self.exp(point_a, 0.5 * V)
