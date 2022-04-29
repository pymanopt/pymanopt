"""Module containing manifolds of n-dimensional rotations."""

import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from scipy.linalg import expm, logm
from scipy.special import comb

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multiprod, multiskew, multisym, multitransp


class SpecialOrthogonalGroup(EuclideanEmbeddedSubmanifold):
    """The special orthogonal group.

    Special orthogonal group (the manifold of rotations): deals with matrices
    X of size k x n x n (or n x n if k = 1, which is the default) such that
    each n x n matrix is orthogonal, with determinant 1, i.e.,
    dot(X.T, X) = eye(n) if k = 1, or dot(X[i].T, X[i]) = eye(n) if k > 1.

    This is a description of SO(n)^k with the induced metric from the
    embedding space (R^nxn)^k, i.e., this manifold is a Riemannian
    submanifold of (R^nxn)^k endowed with the usual trace inner product.

    Tangent vectors are represented in the Lie algebra, i.e., as skew
    symmetric matrices. Use the function manifold.tangent2ambient(X, H) to
    switch from the Lie algebra representation to the embedding space
    representation. This is often necessary when defining
    problem.ehess(X, H).

    By default, the retraction is only a first-order approximation of the
    exponential. To force the use of a second-order approximation, call
    manifold.retr = manifold.retr2 after creating M. This switches from a
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
            name = f"Special orthogonal group SO({n})"
        elif k > 1:
            name = f"Sphecial orthogonal group SO({n})^{k}"
        else:
            raise ValueError("k must be an integer no less than 1.")
        dimension = int(k * comb(n, 2))
        super().__init__(name, dimension)

        if retraction == "qr":
            self.retr = self._retr_qr
        elif retraction == "polar":
            self.retr = self._retr_polar
        else:
            raise ValueError(f"Invalid retraction type '{retraction}'")

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return la.norm(tangent_vector)

    @property
    def typical_dist(self):
        return np.pi * np.sqrt(self._n * self._k)

    def dist(self, point_a, point_b):
        return self.norm(point_a, self.log(point_a, point_b))

    def proj(self, point, vector):
        return multiskew(multiprod(multitransp(point), vector))

    def tangent(self, point, vector):
        return multiskew(vector)

    def tangent2ambient(self, point, tangent_vector):
        return multiprod(point, tangent_vector)

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        Xt = multitransp(point)
        Xtegrad = multiprod(Xt, euclidean_gradient)
        symXtegrad = multisym(Xtegrad)
        Xtehess = multiprod(Xt, euclidean_hvp)
        return multiskew(Xtehess - multiprod(tangent_vector, symXtegrad))

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
            return multiskew(np.real(logm(U)))

        for i in range(self._k):
            U[i] = np.real(logm(U[i]))
        return multiskew(U)

    @staticmethod
    def _randrot(n, N=1):
        if n == 1:
            return np.ones((N, 1, 1))

        R = np.zeros((N, n, n))
        for i in range(N):
            # Generated as such, Q is uniformly distributed over O(n), the
            # group of orthogonal n-by-n matrices.
            A = rnd.randn(n, n)
            Q, RR = la.qr(A)
            # TODO(nkoep): Add a proper reference to Mezzadri 2007.
            Q = Q @ np.diag(np.sign(np.diag(RR)))

            # If Q is in O(n) but not in SO(n), we permute the two first
            # columns of Q such that det(new Q) = -det(Q), hence the new Q will
            # be in SO(n), uniformly distributed.
            if la.det(Q) < 0:
                Q[:, [0, 1]] = Q[:, [1, 0]]
            R[i] = Q

        if N == 1:
            return R.reshape(n, n)
        return R

    def rand(self):
        return self._randrot(self._n, self._k)

    @staticmethod
    def _randskew(n, N=1):
        idxs = np.triu_indices(n, 1)
        S = np.zeros((N, n, n))
        for i in range(N):
            S[i][idxs] = rnd.randn(int(n * (n - 1) / 2))
            S = S - multitransp(S)
        if N == 1:
            return S.reshape(n, n)
        return S

    def randvec(self, point):
        tangent_vector = self._randskew(self._n, self._k)
        return tangent_vector / np.sqrt(
            np.tensordot(
                tangent_vector, tangent_vector, axes=tangent_vector.ndim
            )
        )

    def zerovec(self, point):
        if self._k == 1:
            return np.zeros((self._n, self._n))
        return np.zeros((self._k, self._n, self._n))

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def pair_mean(self, point_a, point_b):
        V = self.log(point_a, point_b)
        return self.exp(point_a, 0.5 * V)
