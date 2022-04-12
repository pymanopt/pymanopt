"""
Module containing manifolds of n-dimensional unitaries
"""

from __future__ import division

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from scipy.linalg import expm, logm
from scipy.special import comb

from pymanopt.tools.multi import multiprod, multihconj, multiherm, multiskewh
from pymanopt.manifolds.manifold import Manifold


class Unitaries(Manifold):
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
    representation. This is often necessary to define problem.ehess(X, H),
    as the input H will then be a skew-Hermitian matrix (but the output must
    not be, as the output is the Hessian in the embedding Euclidean space.)

    By default, the retraction is only a first-order approximation of the
    exponential. To force the use of a second-order approximation, call
    manifold.retr = manifold.retr2 after creating manifold object. This switches from a 
    QR-based computation to an SVD-based computation.

    By default, k = 1.
    
    Example. Based on the example found at:
    http://www.manopt.org/manifold_documentation_rotations.html

    >>> import numpy as np
    >>> from pymanopt import Problem
    >>> from pymanopt.solvers import TrustRegions
    >>> from pymanopt.manifolds import Unitaries

    Generate the problem data.
    >>> n = 3
    >>> m = 10
    >>> A = np.random.randn(n, m)
    >>> B = np.random.randn(n, m)
    >>> ABt = np.dot(A,B.T)

    Create manifold - U(n).
    >>> manifold = Unitaries(n)

    Define the cost function.
    >>> cost = lambda X : -np.tensordot(X.conj(), ABt, axes=X.ndim)

    Define and solve the problem.
    >>> problem = Problem(manifold=manifold, cost=cost)
    >>> solver = TrustRegions()
    >>> X = solver.solve(problem)

    See also: Stiefel Rotations

    This file is based on unitaryfactory from Manopt: www.manopt.org
    Ported by: Haotian Wei
    Original author: Nicolas Boumal, June 18, 2019.
    """

    def __init__(self, n, k=1):
        if k == 1:
            self._name = 'Unitary manifold U({n})'.format(n=n)
        elif k > 1:
            self._name = 'Product unitary manifold U({n})^{k}'.format(n=n, k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._n = n
        self._k = k

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._k * self._n**2

    def inner(self, X, U, V):
        return np.tensordot(U.conj(), V, axes=U.ndim)

    def norm(self, X, U):
        return la.norm(U)

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._n * self._k)

    def proj(self, X, H):
        return multiskewh(multiprod(multihconj(X), H))

    def tangent(self, X, H):
        return multiskewh(H)

    def tangent2ambient(self, X, U):
        return multiprod(X, U)

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        Xt = multihconj(X)
        Xtegrad = multiprod(Xt, egrad)
        symXtegrad = multiherm(Xtegrad)
        Xtehess = multiprod(Xt, ehess)
        return multiskewh(Xtehess - multiprod(H, symXtegrad))

    # QR-based retraction or Polar-based retraction
    # QR-based retraction
    def retr(self, X, U):

        def retri(Y):
            Q, R = la.qr(Y)
            return np.dot(Q, np.diag(np.sign(np.sign(np.diag(R)) + 0.5)))

        Y = X + multiprod(X, U)
        if self._k == 1:
            return retri(Y)
        else:
            for i in range(self._k):
                Y[i] = retri(Y[i])
            return Y

    # Polar-based retraction
    def retr2(self, X, U):

        def retr2i(Y):
            U, _, Vt = la.svd(Y)
            return np.dot(U, Vt)

        Y = X + multiprod(X, U)
        if self._k == 1:
            return retr2i(Y)
        else:
            for i in range(self._k):
                Y[i] = retr2i(Y[i])
        return Y

    def exp(self, X, U):
        expU = U
        if self._k == 1:
            return multiprod(X, expm(expU))
        else:
            for i in range(self._k):
                expU[i] = expm(expU[i])
            return multiprod(X, expU)

    def log(self, X, Y):
        U = multiprod(multihconj(X), Y)
        if self._k == 1:
            return multiskewh(logm(U))
        else:
            for i in range(self._k):
                U[i] = logm(U[i])
        return multiskewh(U)

    # An unused function in current Python version, not enabled temorarily
    # def hash(X):
    #     return 'z {}'.format(
    #         hashlib.md5(np.real(X.reshape(-1)), np.imag(X.reshape(-1))))

    def rand(self):
        return randunitary(self._n, self._k)

    def randvec(self, X):
        U = randskewh(self._n, self._k)
        nrmU = np.sqrt(np.tensordot(U.conj(), U, axes=U.ndim))
        return U / nrmU

    def zerovec(self, X):
        if self._k == 1:
            return np.zeros((self._n, self._n))
        else:
            return np.zeros((self._k, self._n, self._n))

    def transp(self, x1, x2, d):
        return d

    def pairmean(self, X, Y):
        V = self.log(X, Y)
        Y = self.exp(X, 0.5 * V)
        return Y

    def dist(self, x, y):
        return self.norm(x, self.log(x, y))


def randunitary(n, N=1):
    # Generates uniformly random unitary matrices.

    if n == 1:
        U = rnd.randn((N, 1, 1)) + 1j * rnd.randn((N, 1, 1))
        return U / np.abs(U)

    U = np.zeros((N, n, n))

    for i in range(N):
        # Generated as such, Q is uniformly distributed over O(n), the set
        # of orthogonal matrices.
        A = rnd.randn(n, n) + 1j * rnd.randn(n, n)
        Q, RR = la.qr(A)
        U[i] = Q

    if N == 1:
        U = U.reshape(n, n)

    return U


def randskewh(n, N=1):
    # Generate random skew-hermitian matrices with normal entries.
    idxs = np.triu_indices(n, 1)
    S = np.zeros((N, n, n))
    for i in range(N):
        S[i][idxs] = rnd.randn(int(n * (n - 1) / 2))
        S = S - multihconj(S)
    if N == 1:
        return S.reshape(n, n)
    return S
