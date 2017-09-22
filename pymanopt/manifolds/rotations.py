from __future__ import division

# Returns a manifold structure to optimize over rotation matrices.
#
# function M = rotationsfactory(n)
# function M = rotationsfactory(n, k)
#
# Special orthogonal group (the manifold of rotations): deals with matrices
# R of size n x n x k (or n x n if k = 1, which is the default) such that
# each n x n matrix is orthogonal, with determinant 1, i.e., X'*X = eye(n)
# if k = 1, or X(:, :, i)' * X(:, :, i) = eye(n) for i = 1 : k if k > 1.
#
# This is a description of SO(n)^k with the induced metric from the
# embedding space (R^nxn)^k, i.e., this manifold is a Riemannian
# submanifold of (R^nxn)^k endowed with the usual trace inner product.
#
# Tangent vectors are represented in the Lie algebra, i.e., as skew
# symmetric matrices. Use the function M.tangent2ambient(X, H) to switch
# from the Lie algebra representation to the embedding space
# representation. This is often necessary when defining
# problem.ehess(X, H).
#
# By default, the retraction is only a first-order approximation of the
# exponential. To force the use of a second-order approximation, call
# M.retr = M.retr2 after creating M. This switches from a QR-based
# computation to an SVD-based computation.
#
# By default, k = 1.
#
# See also: stiefelfactory

# This file is part of Manopt: www.manopt.org.
# Original author: Nicolas Boumal, Dec. 30, 2012.
# Contributors:
# Change log:
#   Jan. 31, 2013 (NB)
#       Added egrad2rgrad and ehess2rhess
#   Oct. 21, 2016 (NB)
#       Added M.retr2: a second-order retraction based on SVD.

# Ported to pymanopt by Lars Tingelstad. September 2017.

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from scipy.linalg import expm, logm

from pymanopt.tools.multi import multiprod, multitransp, multisym
from pymanopt.manifolds.manifold import Manifold


def randrot(n, N=1):

    if n == 1:
        return np.ones((N, 1, 1))

    R = np.zeros((N, n, n))

    for i in range(N):
        # Generated as such, Q is uniformly distributed over O(n), the set
        # of orthogonal matrices.
        A = rnd.randn(n, n)
        Q, RR = la.qr(A)
        Q = np.dot(Q, np.diag(np.sign(np.diag(RR))))  ## Mezzadri 2007

        # If Q is in O(n) but not in SO(n), we permute the two first
        # columns of Q such that det(new Q) = -det(Q), hence the new Q will
        # be in SO(n), uniformly distributed.
        if la.det(Q) < 0:
            Q[:, [0, 1]] = Q[:, [1, 0]]

        R[i] = Q

    return R


def randskew(n, N=1):
    idxs = np.triu_indices(n, 1)
    S = np.zeros((N, n, n))
    for Si in S:
        Si[idxs] = rnd.randn(int(n * (n - 1) / 2))
    S = S - multitransp(S)
    return S
