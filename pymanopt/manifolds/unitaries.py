"""
Module containing manifolds of n-dimensional unitaries
"""

import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from scipy.linalg import expm, logm
from scipy.special import comb

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multiprod, multihconj, multiherm, multiskewh


class Unitaries(EuclideanEmbeddedSubmanifold):
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

    def __init__(self, n, k=1, retraction="qr"):
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

        if retraction == "qr":
            self.retr = self._retr_qr
        elif retraction == "polar":
            self.retr = self._retr_polar
        else:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a.conj(), tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return la.norm(tangent_vector)

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._n * self._k)

    def dist(self, point_a, point_b):
        return self.norm(point_a, self.log(point_a, point_b))

    def proj(self, point, vector):
        return multiskewh(multiprod(multihconj(point), vector))

    def tangent(self, point, vector):
        return multiskewh(vector)

    def tangent2ambient(self, point, tangent_vector):
        return multiprod(point, tangent_vector)

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        Xt = multihconj(point)
        Xtegrad = multiprod(Xt, euclidean_gradient)
        symXtegrad = multiherm(Xtegrad)
        Xtehess = multiprod(Xt, euclidean_hvp)
        return multiskewh(Xtehess - multiprod(tangent_vector, symXtegrad))

    # QR-based retraction or Polar-based retraction
    # QR-based retraction
    def _retr_qr(self, point, tangent_vector):
        def retri(array):
            q, r = la.qr(array)
            return q @ np.diag(np.sign(np.sign(np.diag(r)) + 0.5))

        Y = point + multiprod(point, tangent_vector)
        if self._k == 1:
            return retri(Y)

        for i in range(self._k):
            Y[i] = retri(Y[i])
        return Y

    # Polar-based retraction
    def _retr_polar(self, point, tangent_vector):
        def retri(array):
            u, _, vt = la.svd(array)
            return u @ vt

        Y = point + multiprod(point, tangent_vector)
        if self._k == 1:
            return retri(Y)

        for i in range(self._k):
            Y[i] = retri(Y[i])
        return Y

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
            return multiskewh(logm(U))

        for i in range(self._k):
            U[i] = logm(U[i])
        return multiskewh(U)

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

    def randvec(self, point):
        tangent_vector = self._randskewh(self._n, self._k)
        nrmU = np.sqrt(np.tensordot(tangent_vector.conj(), tangent_vector, axes=tangent_vector.ndim))
        return tangent_vector / nrmU

    def zerovec(self, point):
        if self._k == 1:
            return np.zeros((self._n, self._n))
        return np.zeros((self._k, self._n, self._n))

    def transp(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pairmean(self, point_a, point_b):
        V = self.log(point_a, point_b)
        return self.exp(point_a, 0.5 * V)