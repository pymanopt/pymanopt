import numpy as np
import scipy.special

from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.tools.multi import (
    multiexpm,
    multilogm,
    multiqr,
    multihconj,
    multiskewh,
    multiherm,
    multitransp,
)


class Unitaries(RiemannianSubmanifold):
    """
    Returns a manifold structure to optimize over unitary matrices.

    manifold = Unitaries(n)
    manifold = Unitaries(n, k)

    Unitary group: deals with arrays U of size n x n x k (or n x n if k = 1,
    which is the default) such that each n x n matrix is unitary, that is,
        X.conj().T @ X = eye(n) if k = 1, or
        X[i].conj().T @ X[i] = eye(n) for i = 1 : k if k > 1.

    This is a description of U(n)^k with the induced metric from the
    embedding space (C^nxn)^k, i.e., this manifold is a Riemannian
    submanifold of (C^nxn)^k endowed with the usual real inner product on
    C^nxn, namely, <A, B> = real(trace(A.conj().T @ B)).

    This is important:
    Tangent vectors are represented in the Lie algebra, i.e., as
    skew-Hermitian matrices. Use the function M.tangent2ambient(X, H) to
    switch from the Lie algebra representation to the embedding space
    representation. This is often necessary when defining
    problem.ehess(X, H).
    as the input H will then be a skew-Hermitian matrix (but the output must
    not be, as the output is the Hessian in the embedding Euclidean space.)

    By default, the retraction is only a first-order approximation of the
    exponential. To force the use of a second-order approximation, call
    manifold.retr = manifold.retr2 after creating manifold object. This switches from a
    QR-based computation to an SVD-based computation.
    Args:
        n: The dimension of the space that elements of the group act on.
        k: The number of elements in the product of groups.
        retraction: The type of retraction to use.
            Possible choices are ``qr`` and ``polar``.
    """

    def __init__(self, n: int, *, k: int = 1, retraction: str = "qr"):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Unitary group U({n})"
        elif k > 1:
            name = f"Product unitary group U({n})^{k}"
        else:
            raise ValueError("k must be an integer no less than 1.")
        dimension = int(k * n**2)
        super().__init__(name, dimension)

        try:
            self._retraction = getattr(self, f"_retraction_{retraction}")
        except AttributeError:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a.conj(), tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    @property
    def typical_dist(self):
        return np.pi * np.sqrt(self._n * self._k)

    def dist(self, point_a, point_b):
        return self.norm(point_a, self.log(point_a, point_b))

    def proj(self, point, vector):
        return multiskewh(multitransp(multihconj(point) @ vector))

    def to_tangent_space(self, point, vector):
        return multiskewh(vector)

    def embedding(self, point, tangent_vector):
        return point @ tangent_vector

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        Xt = multihconj(point)
        Xtegrad = Xt @ euclidean_gradient
        symXtegrad = multiherm(Xtegrad)
        Xtehess = Xt @ euclidean_hessian
        return multiskewh(Xtehess - tangent_vector @ symXtegrad)

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
        return multiskewh(multilogm(multihconj(point_a) @ point_b))

    @staticmethod
    def _randunitary(n, N=1):
        # Generates uniformly random unitary matrices.

        if n == 1:
            U = rnd.randn(N, 1, 1) + 1j * rnd.randn(N, 1, 1)
            if N == 1:
                U = U.reshape(1, 1)
            return U / np.abs(U)

        U = np.zeros((N, n, n), dtype=complex)

        for i in range(N):
            # Generated as such, Q is uniformly distributed over O(n), the set
            # of orthogonal matrices.
            A = rnd.randn(n, n) + 1j * rnd.randn(n, n)
            Q, RR = la.qr(A)
            U[i] = Q

        if N == 1:
            return U.reshape(n, n)
        return U

    def rand(self):
        return self._randunitary(self._n, self._k)

    @staticmethod
    def _randskewh(n, N=1):
        # Generate random skew-hermitian matrices with normal entries.
        idxs = np.triu_indices(n, 1)
        S = np.zeros((N, n, n))
        for i in range(N):
            S[i][idxs] = rnd.randn(int(n * (n - 1) / 2))
            S = S - multihconj(S)
        if N == 1:
            return S.reshape(n, n)
        return S

    def random_tangent_vector(self, point):
        tangent_vector = self._randskewh(self._n, self._k)
        nrmU = np.sqrt(np.tensordot(tangent_vector.conj(),
                       tangent_vector, axes=tangent_vector.ndim))
        return tangent_vector / nrmU

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